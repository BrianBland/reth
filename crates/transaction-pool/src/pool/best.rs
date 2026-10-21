use crate::{
    error::{Eip4844PoolTransactionError, InvalidPoolTransactionError},
    identifier::{SenderId, TransactionId},
    pool::pending::PendingTransaction,
    PoolTransaction, Priority, TransactionOrdering, ValidPoolTransaction,
};
use alloy_consensus::Transaction;
use alloy_primitives::map::AddressSet;
use core::fmt;
use imbl::OrdMap;
use reth_primitives_traits::transaction::error::InvalidTransactionError;
use rustc_hash::FxHashSet;
use std::{
    collections::{BTreeSet, VecDeque},
    sync::Arc,
};
use tokio::sync::broadcast::{error::TryRecvError, Receiver};
use tracing::debug;

const MAX_NEW_TRANSACTIONS_PER_BATCH: usize = 16;

/// The independent transactions of a [`BestTransactions`] iterator, as an ordered set.
///
/// Most of them come from the pool snapshot and are only ever popped, so they are kept in a vector
/// sorted once, which is cheaper to build and pop than a [`BTreeSet`]. Transactions added later go
/// into a [`BTreeSet`].
#[derive(Debug)]
pub(crate) struct IndependentTransactions<T: TransactionOrdering> {
    /// Snapshot transactions, sorted ascending so the best is last.
    snapshot: Vec<PendingTransaction<T>>,
    /// Transactions inserted after the snapshot.
    added: BTreeSet<PendingTransaction<T>>,
}

impl<T: TransactionOrdering> IndependentTransactions<T> {
    /// Returns the set of the given snapshot transactions, which must be distinct.
    pub(crate) fn new(mut snapshot: Vec<PendingTransaction<T>>) -> Self {
        snapshot.sort_unstable();
        Self { snapshot, added: BTreeSet::new() }
    }

    /// Inserts a transaction, unless it is already in the set.
    ///
    /// The same transaction can be inserted again while it is still in the snapshot, e.g. when a
    /// reinserted ancestor is yielded and unlocks it. Like a single `BTreeSet`, keep one copy.
    pub(crate) fn insert(&mut self, transaction: PendingTransaction<T>) {
        if self.snapshot.binary_search(&transaction).is_err() {
            self.added.insert(transaction);
        }
    }

    /// Removes and returns the best transaction.
    pub(crate) fn pop_last(&mut self) -> Option<PendingTransaction<T>> {
        match (self.snapshot.last(), self.added.last()) {
            (Some(snapshot), Some(added)) if added > snapshot => self.added.pop_last(),
            (Some(_), _) => self.snapshot.pop(),
            (None, _) => self.added.pop_last(),
        }
    }

    /// Returns the number of transactions.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.snapshot.len() + self.added.len()
    }

    /// Returns whether the set contains the transaction.
    #[cfg(test)]
    pub(crate) fn contains(&self, transaction: &PendingTransaction<T>) -> bool {
        self.added.contains(transaction) || self.snapshot.contains(transaction)
    }

    /// Returns all transactions, in no particular order.
    #[cfg(test)]
    pub(crate) fn iter(&self) -> impl Iterator<Item = &PendingTransaction<T>> {
        self.snapshot.iter().chain(&self.added)
    }
}

/// An iterator that returns transactions that can be executed on the current state (*best*
/// transactions).
///
/// This is a wrapper around [`BestTransactions`] that also enforces a specific basefee.
///
/// This iterator guarantees that all transactions it returns satisfy both the base fee and blob
/// fee!
pub(crate) struct BestTransactionsWithFees<T: TransactionOrdering> {
    pub(crate) best: BestTransactions<T>,
    pub(crate) base_fee: u64,
    pub(crate) base_fee_per_blob_gas: u64,
}

impl<T: TransactionOrdering> crate::traits::BestTransactions for BestTransactionsWithFees<T> {
    fn mark_invalid(&mut self, tx: &Self::Item, kind: InvalidPoolTransactionError) {
        BestTransactions::mark_invalid(&mut self.best, tx, kind)
    }

    fn no_updates(&mut self) {
        self.best.no_updates()
    }

    fn allow_updates_out_of_order(&mut self) {
        self.best.allow_updates_out_of_order()
    }

    fn skip_blobs(&mut self) {
        self.set_skip_blobs(true)
    }

    fn set_skip_blobs(&mut self, skip_blobs: bool) {
        self.best.set_skip_blobs(skip_blobs)
    }
}

impl<T: TransactionOrdering> Iterator for BestTransactionsWithFees<T> {
    type Item = Arc<ValidPoolTransaction<T::Transaction>>;

    fn next(&mut self) -> Option<Self::Item> {
        // find the next transaction that satisfies the base fee
        loop {
            let best = Iterator::next(&mut self.best)?;
            // If both the base fee and blob fee (if applicable for EIP-4844) are satisfied, return
            // the transaction
            if best.transaction.max_fee_per_gas() >= self.base_fee as u128 &&
                best.transaction
                    .max_fee_per_blob_gas()
                    .is_none_or(|fee| fee >= self.base_fee_per_blob_gas as u128)
            {
                return Some(best);
            }
            crate::traits::BestTransactions::mark_invalid(
                self,
                &best,
                InvalidPoolTransactionError::Underpriced,
            );
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.best.size_hint();
        (0, upper)
    }
}

/// An iterator that returns transactions that can be executed on the current state (*best*
/// transactions).
///
/// The [`PendingPool`](crate::pool::pending::PendingPool) contains transactions that *could* all
/// be executed on the current state, but only yields transactions that are ready to be executed
/// now. While it contains all gapless transactions of a sender, it _always_ only returns the
/// transaction with the current on chain nonce.
#[derive(Debug)]
pub struct BestTransactions<T: TransactionOrdering> {
    /// Contains a copy of _all_ transactions of the pending pool at the point in time this
    /// iterator was created.
    pub(crate) all: OrdMap<TransactionId, PendingTransaction<T>>,
    /// Transactions in `all` that were already yielded, and so are logically removed.
    ///
    /// `all` is a structural-sharing clone of the pool, so removing from it copies the path to
    /// each removed entry. Tombstoning yielded ids here keeps `all` unmodified on the hot path.
    pub(crate) yielded: FxHashSet<TransactionId>,
    /// Number of tombstones in `yielded` above which they are removed from `all` in bulk: the
    /// size of the snapshot. Draining the snapshot never removes anything, while an iterator that
    /// keeps yielding live updates retains at most this many yielded transactions.
    pub(crate) max_yielded: usize,
    /// Whether a live update replaced a transaction already in `all`, which can leave a stale
    /// copy of it in `independent`.
    pub(crate) replaced: bool,
    /// Transactions that can be executed right away: these have the expected nonce.
    ///
    /// Once an `independent` transaction with the nonce `N` is returned, it unlocks `N+1`, which
    /// then can be moved from the `all` set to the `independent` set.
    pub(crate) independent: IndependentTransactions<T>,
    /// There might be the case where a yielded transactions is invalid, this will track it.
    pub(crate) invalid: FxHashSet<SenderId>,
    /// Used to receive any new pending transactions that have been added to the pool after this
    /// iterator was static filtered
    ///
    /// These new pending transactions are inserted into this iterator's pool before yielding the
    /// next value
    pub(crate) new_transaction_receiver: Option<Receiver<PendingTransaction<T>>>,
    /// The priority value of most recently yielded transaction.
    ///
    /// This is required if new pending transactions are fed in while it yields new values.
    pub(crate) last_priority: Option<Priority<T::PriorityValue>>,
    /// Flag to control whether to skip blob transactions (EIP4844).
    pub(crate) skip_blobs: bool,
    /// Whether live updates can be yielded after a lower-priority transaction.
    pub(crate) allow_updates_out_of_order: bool,
}

impl<T: TransactionOrdering> BestTransactions<T> {
    /// Mark the transaction and its descendants as invalid.
    pub(crate) fn mark_invalid(
        &mut self,
        tx: &Arc<ValidPoolTransaction<T::Transaction>>,
        _kind: InvalidPoolTransactionError,
    ) {
        self.invalid.insert(tx.sender_id());
    }

    /// Returns the ancestor the given transaction, the transaction with `nonce - 1`.
    ///
    /// Note: for a transaction with nonce higher than the current on chain nonce this will always
    /// return an ancestor since all transactions in this pool are gapless.
    pub(crate) fn ancestor(&self, id: &TransactionId) -> Option<&PendingTransaction<T>> {
        self.get(&id.unchecked_ancestor()?)
    }

    /// Returns a transaction from `all` that has not been yielded yet.
    fn get(&self, id: &TransactionId) -> Option<&PendingTransaction<T>> {
        self.all.get(id).filter(|_| !self.yielded.contains(id))
    }

    /// Inserts a transaction into `all`, replacing any yielded entry with the same id.
    fn insert(&mut self, id: TransactionId, tx: PendingTransaction<T>) {
        self.yielded.remove(&id);
        self.replaced |= self.all.insert(id, tx).is_some();
    }

    /// Non-blocking read on the new pending transactions subscription channel
    fn try_recv(&mut self) -> Option<IncomingTransaction<T>> {
        loop {
            match self.new_transaction_receiver.as_mut()?.try_recv() {
                Ok(tx) => {
                    if !self.allow_updates_out_of_order &&
                        let Some(last_priority) = &self.last_priority &&
                        &tx.priority > last_priority
                    {
                        // we skip transactions if we already yielded a transaction with lower
                        // priority
                        return Some(IncomingTransaction::Stash(tx))
                    }
                    return Some(IncomingTransaction::Process(tx))
                }
                // note TryRecvError::Lagged can be returned here, which is an error that attempts
                // to correct itself on consecutive try_recv() attempts

                // the cost of ignoring this error is allowing old transactions to get
                // overwritten after the chan buffer size is met
                Err(TryRecvError::Lagged(_)) => {
                    // Handle the case where the receiver lagged too far behind.
                    // `num_skipped` indicates the number of messages that were skipped.
                }

                // this case is still better than the existing iterator behavior where no new
                // pending txs are surfaced to consumers
                Err(_) => return None,
            }
        }
    }

    /// Removes the currently best independent transaction from the independent set and the total
    /// set.
    fn pop_best(&mut self) -> Option<PendingTransaction<T>> {
        loop {
            let best = self.independent.pop_last()?;
            let id = *best.transaction.id();
            // After a replacement, an entry whose id is no longer live (already yielded, possibly
            // compacted out of `all`) is a stale copy, e.g. the snapshot entry of a transaction
            // that was replaced and yielded through a live update. Skip it so it is neither
            // yielded twice nor tombstoned outside `all`.
            if self.replaced && self.get(&id).is_none() {
                continue
            }
            self.yielded.insert(id);
            if self.yielded.len() > self.max_yielded {
                for id in self.yielded.drain() {
                    self.all.remove(&id);
                }
            }
            return Some(best)
        }
    }

    /// Checks for new transactions that have come into the `PendingPool` after this iterator was
    /// created and inserts them
    fn add_new_transactions(&mut self) {
        for _ in 0..MAX_NEW_TRANSACTIONS_PER_BATCH {
            if let Some(pending_tx) = self.try_recv() {
                //  same logic as PendingPool::add_transaction/PendingPool::best_with_unlocked

                match pending_tx {
                    IncomingTransaction::Process(tx) => {
                        let tx_id = *tx.transaction.id();
                        if self.ancestor(&tx_id).is_none() {
                            self.independent.insert(tx.clone());
                        }
                        self.insert(tx_id, tx);
                    }
                    IncomingTransaction::Stash(tx) => {
                        let tx_id = *tx.transaction.id();
                        self.insert(tx_id, tx);
                    }
                }
            } else {
                break;
            }
        }
    }

    /// Returns the next best transaction and its priority value.
    #[expect(clippy::type_complexity)]
    pub fn next_tx_and_priority(
        &mut self,
    ) -> Option<(Arc<ValidPoolTransaction<T::Transaction>>, Priority<T::PriorityValue>)> {
        loop {
            self.add_new_transactions();
            // Remove the next independent tx with the highest priority
            let best = self.pop_best()?;
            let sender_id = best.transaction.sender_id();

            // skip transactions for which sender was marked as invalid
            if self.invalid.contains(&sender_id) {
                debug!(
                    target: "txpool",
                    "[{:?}] skipping invalid transaction",
                    best.transaction.hash()
                );
                continue
            }

            // Insert transactions that just got unlocked.
            if let Some(unlocked) = self.get(&best.unlocks()) {
                self.independent.insert(unlocked.clone());
            }

            if self.skip_blobs && best.transaction.is_eip4844() {
                // blobs should be skipped, marking them as invalid will ensure that no dependent
                // transactions are returned
                self.mark_invalid(
                    &best.transaction,
                    InvalidPoolTransactionError::Eip4844(
                        Eip4844PoolTransactionError::NoEip4844Blobs,
                    ),
                )
            } else {
                if self.new_transaction_receiver.is_some() {
                    self.last_priority = Some(best.priority.clone())
                }
                return Some((best.transaction, best.priority))
            }
        }
    }
}

/// Result of attempting to receive a new transaction from the channel during iteration.
///
/// This enum determines how a newly received transaction should be handled based on its priority
/// relative to transactions already yielded by the iterator.
enum IncomingTransaction<T: TransactionOrdering> {
    /// Process the transaction normally: add to both `all` map and potentially to `independent`
    /// set (if it has no ancestor).
    ///
    /// This variant is used when the transaction's priority is lower than or equal to the last
    /// yielded transaction, meaning it can be safely processed without breaking the descending
    /// priority order.
    Process(PendingTransaction<T>),

    /// Stash the transaction: add only to the `all` map, but NOT to the `independent` set.
    ///
    /// This variant is used when the transaction has a higher priority than the last yielded
    /// transaction. We cannot yield it immediately (to maintain strict priority ordering), but we
    /// must still track it so that:
    /// - Its descendants can find it via `ancestor()` lookups
    /// - We prevent those descendants from being incorrectly promoted to `independent`
    ///
    /// Without stashing, if a child of this transaction arrives later, it would fail to find its
    /// parent in `all`, be marked as `independent`, and be yielded out of order (before its
    /// parent), causing nonce gaps.
    Stash(PendingTransaction<T>),
}

impl<T: TransactionOrdering> crate::traits::BestTransactions for BestTransactions<T> {
    fn mark_invalid(&mut self, tx: &Self::Item, kind: InvalidPoolTransactionError) {
        Self::mark_invalid(self, tx, kind)
    }

    fn no_updates(&mut self) {
        self.new_transaction_receiver.take();
        self.last_priority.take();
    }

    fn allow_updates_out_of_order(&mut self) {
        self.allow_updates_out_of_order = true;
    }

    fn skip_blobs(&mut self) {
        self.set_skip_blobs(true);
    }

    fn set_skip_blobs(&mut self, skip_blobs: bool) {
        self.skip_blobs = skip_blobs;
    }
}

impl<T: TransactionOrdering> Iterator for BestTransactions<T> {
    type Item = Arc<ValidPoolTransaction<T::Transaction>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_tx_and_priority().map(|(tx, _)| tx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.new_transaction_receiver.is_none().then_some(self.all.len() - self.yielded.len()))
    }
}

/// A [`BestTransactions`](crate::traits::BestTransactions) implementation that filters the
/// transactions of iter with predicate.
///
/// Filter out transactions are marked as invalid:
/// [`BestTransactions::mark_invalid`](crate::traits::BestTransactions::mark_invalid).
pub struct BestTransactionFilter<I, P> {
    pub(crate) best: I,
    pub(crate) predicate: P,
}

impl<I, P> BestTransactionFilter<I, P> {
    /// Create a new [`BestTransactionFilter`] with the given predicate.
    pub const fn new(best: I, predicate: P) -> Self {
        Self { best, predicate }
    }
}

impl<I, P> Iterator for BestTransactionFilter<I, P>
where
    I: crate::traits::BestTransactions,
    P: FnMut(&<I as Iterator>::Item) -> bool,
{
    type Item = <I as Iterator>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let best = self.best.next()?;
            if (self.predicate)(&best) {
                return Some(best)
            }
            self.best.mark_invalid(
                &best,
                InvalidPoolTransactionError::Consensus(InvalidTransactionError::TxTypeNotSupported),
            );
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.best.size_hint();
        (0, upper)
    }
}

impl<I, P> crate::traits::BestTransactions for BestTransactionFilter<I, P>
where
    I: crate::traits::BestTransactions,
    P: FnMut(&<I as Iterator>::Item) -> bool + Send,
{
    fn mark_invalid(&mut self, tx: &Self::Item, kind: InvalidPoolTransactionError) {
        crate::traits::BestTransactions::mark_invalid(&mut self.best, tx, kind)
    }

    fn no_updates(&mut self) {
        self.best.no_updates()
    }

    fn allow_updates_out_of_order(&mut self) {
        self.best.allow_updates_out_of_order()
    }

    fn skip_blobs(&mut self) {
        self.set_skip_blobs(true)
    }

    fn set_skip_blobs(&mut self, skip_blobs: bool) {
        self.best.set_skip_blobs(skip_blobs)
    }
}

impl<I: fmt::Debug, P> fmt::Debug for BestTransactionFilter<I, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BestTransactionFilter").field("best", &self.best).finish()
    }
}

/// Wrapper over [`crate::traits::BestTransactions`] that prioritizes transactions of certain
/// senders capping total gas used by such transactions.
#[derive(Debug)]
pub struct BestTransactionsWithPrioritizedSenders<I: Iterator> {
    /// Inner iterator
    inner: I,
    /// A set of senders which transactions should be prioritized
    prioritized_senders: AddressSet,
    /// Maximum total gas limit of prioritized transactions
    max_prioritized_gas: u64,
    /// Buffer with transactions that are not being prioritized. Those will be the first to be
    /// included after the prioritized transactions
    buffer: VecDeque<I::Item>,
    /// Tracker of total gas limit of prioritized transactions. Once it reaches
    /// `max_prioritized_gas` no more transactions will be prioritized
    prioritized_gas: u64,
}

impl<I: Iterator> BestTransactionsWithPrioritizedSenders<I> {
    /// Constructs a new [`BestTransactionsWithPrioritizedSenders`].
    pub fn new(prioritized_senders: AddressSet, max_prioritized_gas: u64, inner: I) -> Self {
        Self {
            inner,
            prioritized_senders,
            max_prioritized_gas,
            buffer: Default::default(),
            prioritized_gas: Default::default(),
        }
    }
}

impl<I, T> Iterator for BestTransactionsWithPrioritizedSenders<I>
where
    I: crate::traits::BestTransactions<Item = Arc<ValidPoolTransaction<T>>>,
    T: PoolTransaction,
{
    type Item = <I as Iterator>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        // If we have space, try prioritizing transactions
        if self.prioritized_gas < self.max_prioritized_gas {
            for item in &mut self.inner {
                if self.prioritized_senders.contains(&item.transaction.sender()) &&
                    self.prioritized_gas + item.transaction.gas_limit() <=
                        self.max_prioritized_gas
                {
                    self.prioritized_gas += item.transaction.gas_limit();
                    return Some(item)
                }
                self.buffer.push_back(item);
            }
        }

        if let Some(item) = self.buffer.pop_front() {
            Some(item)
        } else {
            self.inner.next()
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let buffered = self.buffer.len();
        let (inner_lower, inner_upper) = self.inner.size_hint();

        (
            buffered.saturating_add(inner_lower),
            inner_upper.and_then(|upper| upper.checked_add(buffered)),
        )
    }
}

impl<I, T> crate::traits::BestTransactions for BestTransactionsWithPrioritizedSenders<I>
where
    I: crate::traits::BestTransactions<Item = Arc<ValidPoolTransaction<T>>>,
    T: PoolTransaction,
{
    fn mark_invalid(&mut self, tx: &Self::Item, kind: InvalidPoolTransactionError) {
        self.inner.mark_invalid(tx, kind)
    }

    fn no_updates(&mut self) {
        self.inner.no_updates()
    }

    fn allow_updates_out_of_order(&mut self) {
        self.inner.allow_updates_out_of_order()
    }

    fn set_skip_blobs(&mut self, skip_blobs: bool) {
        if skip_blobs {
            self.buffer.retain(|tx| !tx.transaction.is_eip4844())
        }
        self.inner.set_skip_blobs(skip_blobs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        pool::pending::PendingPool,
        test_utils::{MockOrdering, MockTransaction, MockTransactionFactory},
        BestTransactions, Priority,
    };

    #[test]
    fn test_best_iter() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        let num_tx = 10;
        // insert 10 gapless tx
        let tx = MockTransaction::eip1559();
        for nonce in 0..num_tx {
            let tx = tx.clone().rng_hash().with_nonce(nonce);
            let valid_tx = f.validated(tx);
            pool.add_transaction(Arc::new(valid_tx), 0);
        }

        let mut best = pool.best();
        assert_eq!(best.all.len(), num_tx as usize);
        assert_eq!(best.independent.len(), 1);

        // check tx are returned in order
        for nonce in 0..num_tx {
            assert_eq!(best.independent.len(), 1);
            let tx = best.next().unwrap();
            assert_eq!(tx.nonce(), nonce);
        }
    }

    #[test]
    fn test_best_transactions_size_hint() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        for nonce in 0..3 {
            let tx = MockTransaction::eip1559().rng_hash().with_nonce(nonce);
            pool.add_transaction(Arc::new(f.validated(tx)), 0);
        }

        let mut best = pool.best();
        assert_eq!(best.size_hint(), (0, None));

        best.no_updates();
        assert_eq!(best.size_hint(), (0, Some(3)));

        assert_eq!(best.next().unwrap().nonce(), 0);
        assert_eq!(best.size_hint(), (0, Some(2)));
    }

    #[test]
    fn test_best_transactions_with_fees_size_hint() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        for nonce in 0..3 {
            let tx = MockTransaction::eip1559().rng_hash().with_nonce(nonce).with_max_fee(100);
            pool.add_transaction(Arc::new(f.validated(tx)), 0);
        }

        let mut best = pool.best_with_basefee_and_blobfee(10, 0);
        best.no_updates();

        assert_eq!(best.size_hint(), (0, Some(3)));
        assert_eq!(best.next().unwrap().nonce(), 0);
        assert_eq!(best.size_hint(), (0, Some(2)));
    }

    #[test]
    fn test_best_transaction_filter_size_hint() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        for nonce in 0..3 {
            let tx = MockTransaction::eip1559().rng_hash().with_nonce(nonce);
            pool.add_transaction(Arc::new(f.validated(tx)), 0);
        }

        let best = pool.best().without_updates();
        let mut filter =
            BestTransactionFilter::new(best, |_: &Arc<ValidPoolTransaction<MockTransaction>>| {
                false
            });

        assert_eq!(filter.size_hint(), (0, Some(3)));
        assert!(filter.next().is_none());
        assert_eq!(filter.size_hint(), (0, Some(0)));
    }

    #[test]
    fn test_best_transactions_with_prioritized_senders_size_hint() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        for gas_price in 0..5 {
            let tx = MockTransaction::eip1559().with_gas_price((gas_price + 1) * 10);
            pool.add_transaction(Arc::new(f.validated(tx)), 0);
        }

        let prioritized_tx = MockTransaction::eip1559().with_gas_price(5).with_gas_limit(200);
        let prioritized_sender = prioritized_tx.sender();
        pool.add_transaction(Arc::new(f.validated(prioritized_tx)), 0);

        let mut best = BestTransactionsWithPrioritizedSenders::new(
            AddressSet::from_iter([prioritized_sender]),
            200,
            pool.best().without_updates(),
        );

        assert_eq!(best.size_hint(), (0, Some(6)));
        assert_eq!(best.next().unwrap().sender(), prioritized_sender);
        assert_eq!(best.size_hint(), (5, Some(5)));
    }

    #[test]
    fn test_best_iter_invalid() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        let num_tx = 10;
        // insert 10 gapless tx
        let tx = MockTransaction::eip1559();
        for nonce in 0..num_tx {
            let tx = tx.clone().rng_hash().with_nonce(nonce);
            let valid_tx = f.validated(tx);
            pool.add_transaction(Arc::new(valid_tx), 0);
        }

        let mut best = pool.best();

        // mark the first tx as invalid
        let invalid = best.independent.iter().next().unwrap();
        best.mark_invalid(
            &invalid.transaction.clone(),
            InvalidPoolTransactionError::Consensus(InvalidTransactionError::TxTypeNotSupported),
        );

        // iterator is empty
        assert!(best.next().is_none());
    }

    #[test]
    fn test_best_transactions_iter_invalid() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        let num_tx = 10;
        // insert 10 gapless tx
        let tx = MockTransaction::eip1559();
        for nonce in 0..num_tx {
            let tx = tx.clone().rng_hash().with_nonce(nonce);
            let valid_tx = f.validated(tx);
            pool.add_transaction(Arc::new(valid_tx), 0);
        }

        let mut best: Box<
            dyn crate::traits::BestTransactions<Item = Arc<ValidPoolTransaction<MockTransaction>>>,
        > = Box::new(pool.best());

        let tx = Iterator::next(&mut best).unwrap();
        crate::traits::BestTransactions::mark_invalid(
            &mut *best,
            &tx,
            InvalidPoolTransactionError::Consensus(InvalidTransactionError::TxTypeNotSupported),
        );
        assert!(Iterator::next(&mut best).is_none());
    }

    #[test]
    fn test_best_with_fees_iter_base_fee_satisfied() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        let num_tx = 5;
        let base_fee: u64 = 10;
        let base_fee_per_blob_gas: u64 = 15;

        // Insert transactions with a max_fee_per_gas greater than or equal to the base fee
        // Without blob fee
        for nonce in 0..num_tx {
            let tx = MockTransaction::eip1559()
                .rng_hash()
                .with_nonce(nonce)
                .with_max_fee(base_fee as u128 + 5);
            let valid_tx = f.validated(tx);
            pool.add_transaction(Arc::new(valid_tx), 0);
        }

        let mut best = pool.best_with_basefee_and_blobfee(base_fee, base_fee_per_blob_gas);

        for nonce in 0..num_tx {
            let tx = best.next().expect("Transaction should be returned");
            assert_eq!(tx.nonce(), nonce);
            assert!(tx.transaction.max_fee_per_gas() >= base_fee as u128);
        }
    }

    #[test]
    fn test_best_with_fees_iter_base_fee_violated() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        let num_tx = 5;
        let base_fee: u64 = 20;
        let base_fee_per_blob_gas: u64 = 15;

        // Insert transactions with a max_fee_per_gas less than the base fee
        for nonce in 0..num_tx {
            let tx = MockTransaction::eip1559()
                .rng_hash()
                .with_nonce(nonce)
                .with_max_fee(base_fee as u128 - 5);
            let valid_tx = f.validated(tx);
            pool.add_transaction(Arc::new(valid_tx), 0);
        }

        let mut best = pool.best_with_basefee_and_blobfee(base_fee, base_fee_per_blob_gas);

        // No transaction should be returned since all violate the base fee
        assert!(best.next().is_none());
    }

    #[test]
    fn test_best_with_fees_iter_blob_fee_satisfied() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        let num_tx = 5;
        let base_fee: u64 = 10;
        let base_fee_per_blob_gas: u64 = 20;

        // Insert transactions with a max_fee_per_blob_gas greater than or equal to the base fee per
        // blob gas
        for nonce in 0..num_tx {
            let tx = MockTransaction::eip4844()
                .rng_hash()
                .with_nonce(nonce)
                .with_max_fee(base_fee as u128 + 5)
                .with_blob_fee(base_fee_per_blob_gas as u128 + 5);
            let valid_tx = f.validated(tx);
            pool.add_transaction(Arc::new(valid_tx), 0);
        }

        let mut best = pool.best_with_basefee_and_blobfee(base_fee, base_fee_per_blob_gas);

        // All transactions should be returned in order since they satisfy both base fee and blob
        // fee
        for nonce in 0..num_tx {
            let tx = best.next().expect("Transaction should be returned");
            assert_eq!(tx.nonce(), nonce);
            assert!(tx.transaction.max_fee_per_gas() >= base_fee as u128);
            assert!(
                tx.transaction.max_fee_per_blob_gas().unwrap() >= base_fee_per_blob_gas as u128
            );
        }

        // No more transactions should be returned
        assert!(best.next().is_none());
    }

    #[test]
    fn test_best_with_fees_iter_blob_fee_violated() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        let num_tx = 5;
        let base_fee: u64 = 10;
        let base_fee_per_blob_gas: u64 = 20;

        // Insert transactions with a max_fee_per_blob_gas less than the base fee per blob gas
        for nonce in 0..num_tx {
            let tx = MockTransaction::eip4844()
                .rng_hash()
                .with_nonce(nonce)
                .with_max_fee(base_fee as u128 + 5)
                .with_blob_fee(base_fee_per_blob_gas as u128 - 5);
            let valid_tx = f.validated(tx);
            pool.add_transaction(Arc::new(valid_tx), 0);
        }

        let mut best = pool.best_with_basefee_and_blobfee(base_fee, base_fee_per_blob_gas);

        // No transaction should be returned since all violate the blob fee
        assert!(best.next().is_none());
    }

    #[test]
    fn test_best_with_fees_iter_mixed_fees() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        let base_fee: u64 = 10;
        let base_fee_per_blob_gas: u64 = 20;

        // Insert transactions with varying max_fee_per_gas and max_fee_per_blob_gas
        let tx1 =
            MockTransaction::eip1559().rng_hash().with_nonce(0).with_max_fee(base_fee as u128 + 5);
        let tx2 = MockTransaction::eip4844()
            .rng_hash()
            .with_nonce(1)
            .with_max_fee(base_fee as u128 + 5)
            .with_blob_fee(base_fee_per_blob_gas as u128 + 5);
        let tx3 = MockTransaction::eip4844()
            .rng_hash()
            .with_nonce(2)
            .with_max_fee(base_fee as u128 + 5)
            .with_blob_fee(base_fee_per_blob_gas as u128 - 5);
        let tx4 =
            MockTransaction::eip1559().rng_hash().with_nonce(3).with_max_fee(base_fee as u128 - 5);

        pool.add_transaction(Arc::new(f.validated(tx1.clone())), 0);
        pool.add_transaction(Arc::new(f.validated(tx2.clone())), 0);
        pool.add_transaction(Arc::new(f.validated(tx3)), 0);
        pool.add_transaction(Arc::new(f.validated(tx4)), 0);

        let mut best = pool.best_with_basefee_and_blobfee(base_fee, base_fee_per_blob_gas);

        let expected_order = vec![tx1, tx2];
        for expected_tx in expected_order {
            let tx = best.next().expect("Transaction should be returned");
            assert_eq!(tx.transaction, expected_tx);
        }

        // No more transactions should be returned
        assert!(best.next().is_none());
    }

    #[test]
    fn test_best_add_transaction_with_next_nonce() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        // Add 5 transactions with increasing nonces to the pool
        let num_tx = 5;
        let tx = MockTransaction::eip1559();
        for nonce in 0..num_tx {
            let tx = tx.clone().rng_hash().with_nonce(nonce);
            let valid_tx = f.validated(tx);
            pool.add_transaction(Arc::new(valid_tx), 0);
        }

        // Create a BestTransactions iterator from the pool
        let mut best = pool.best();

        // Use a broadcast channel for transaction updates
        let (tx_sender, tx_receiver) =
            tokio::sync::broadcast::channel::<PendingTransaction<MockOrdering>>(1000);
        best.new_transaction_receiver = Some(tx_receiver);

        // Create a new transaction with nonce 5 and validate it
        let new_tx = MockTransaction::eip1559().rng_hash().with_nonce(5);
        let valid_new_tx = f.validated(new_tx);

        // Send the new transaction through the broadcast channel
        let pending_tx = PendingTransaction {
            submission_id: 10,
            transaction: Arc::new(valid_new_tx.clone()),
            priority: Priority::Value(1000),
        };
        tx_sender.send(pending_tx.clone()).unwrap();

        // Add new transactions to the iterator
        best.add_new_transactions();

        // Verify that the new transaction has been added to the 'all' map
        assert_eq!(best.all.len(), 6);
        assert!(best.all.contains_key(valid_new_tx.id()));

        // Verify that the new transaction has been added to the 'independent' set
        assert_eq!(best.independent.len(), 2);
        assert!(best.independent.contains(&pending_tx));
    }

    #[test]
    fn descendant_arriving_after_its_yielded_ancestor_is_independent() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();
        let parent = f.validated(MockTransaction::eip1559().rng_hash().with_nonce(0));
        pool.add_transaction(Arc::new(parent.clone()), 0);

        let mut best = pool.best();
        let (tx_sender, tx_receiver) =
            tokio::sync::broadcast::channel::<PendingTransaction<MockOrdering>>(10);
        best.new_transaction_receiver = Some(tx_receiver);
        assert_eq!(best.next().unwrap().id(), parent.id());

        let child = f.validated(parent.transaction.rng_hash().with_nonce(1));
        tx_sender
            .send(PendingTransaction {
                submission_id: 1,
                transaction: Arc::new(child.clone()),
                priority: Priority::Value(0),
            })
            .unwrap();
        assert_eq!(best.next().unwrap().id(), child.id());
        assert!(best.next().is_none());
    }

    #[test]
    fn reinserted_ancestor_does_not_duplicate_its_snapshot_descendant() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();
        // A nonce-1 transaction whose ancestor is not in the pool is independent in the snapshot.
        let child = f.validated(
            MockTransaction::eip1559()
                .rng_hash()
                .with_nonce(1)
                .with_priority_fee(100)
                .with_max_fee(100),
        );
        pool.add_transaction(Arc::new(child.clone()), 0);

        let mut best = pool.best();
        let (tx_sender, tx_receiver) =
            tokio::sync::broadcast::channel::<PendingTransaction<MockOrdering>>(10);
        best.new_transaction_receiver = Some(tx_receiver);

        // The ancestor is reinserted (e.g. after a reorg) at a higher priority. Yielding it unlocks
        // the child, which is still in the snapshot set.
        let parent = f.validated(child.transaction.clone().rng_hash().with_nonce(0));
        tx_sender
            .send(PendingTransaction {
                submission_id: 1,
                transaction: Arc::new(parent.clone()),
                priority: Priority::Value(1_000),
            })
            .unwrap();

        let yielded: Vec<_> = std::iter::from_fn(|| best.next()).map(|tx| *tx.hash()).collect();
        assert!(best.yielded.iter().all(|id| best.all.contains_key(id)));
        let _ = best.size_hint();
        assert_eq!(yielded, vec![*parent.hash(), *child.hash()]);
    }

    #[test]
    fn independent_transactions_pop_snapshot_and_added_in_priority_order() {
        let mut f = MockTransactionFactory::default();
        let mut pending = |priority: u128| PendingTransaction::<MockOrdering> {
            submission_id: 0,
            transaction: Arc::new(f.validated(MockTransaction::eip1559().rng_hash())),
            priority: Priority::Value(priority),
        };
        let (p1, p2, p3, p4) = (pending(1), pending(2), pending(3), pending(4));
        let mut independent = IndependentTransactions::new(vec![p3.clone(), p1.clone()]);
        independent.insert(p4.clone());
        independent.insert(p2.clone());

        let popped: Vec<_> = std::iter::from_fn(|| independent.pop_last()).collect();
        assert_eq!(popped, vec![p4, p3, p2, p1]);
    }

    #[test]
    fn independent_transactions_break_equal_priority_ties_by_submission_across_stores() {
        let mut f = MockTransactionFactory::default();
        let mut pending = |submission_id: u64| PendingTransaction::<MockOrdering> {
            submission_id,
            transaction: Arc::new(f.validated(MockTransaction::eip1559().rng_hash())),
            priority: Priority::Value(5),
        };
        let (oldest, middle, newest) = (pending(1), pending(2), pending(3));
        let mut independent = IndependentTransactions::new(vec![oldest.clone(), newest.clone()]);
        independent.insert(middle.clone());

        // Equal priorities yield the earliest submission first, whichever store holds it.
        let popped: Vec<_> = std::iter::from_fn(|| independent.pop_last()).collect();
        assert_eq!(popped, vec![oldest, middle, newest]);
    }

    #[test]
    fn replaced_descendant_is_yielded_once_and_counted_after_no_updates() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();
        let parent = f.validated(MockTransaction::eip1559().rng_hash().with_nonce(0));
        let child = f.validated(parent.transaction.clone().rng_hash().with_nonce(1));
        pool.add_transaction(Arc::new(parent.clone()), 0);
        pool.add_transaction(Arc::new(child.clone()), 0);

        let mut best = pool.best();
        let (tx_sender, tx_receiver) =
            tokio::sync::broadcast::channel::<PendingTransaction<MockOrdering>>(10);
        best.new_transaction_receiver = Some(tx_receiver);
        let replacement = f.validated(child.transaction.clone().rng_hash().inc_price());
        assert_eq!(replacement.id(), child.id());
        tx_sender
            .send(PendingTransaction {
                submission_id: 2,
                transaction: Arc::new(replacement.clone()),
                priority: Priority::Value(0),
            })
            .unwrap();

        assert_eq!(best.next().unwrap().hash(), parent.hash());
        best.no_updates();
        assert_eq!(best.size_hint(), (0, Some(1)));
        assert_eq!(best.next().unwrap().hash(), replacement.hash());
        assert!(best.next().is_none());
        assert_eq!(best.size_hint(), (0, Some(0)));
    }

    #[test]
    fn descendant_arriving_before_its_lone_ancestor_is_yielded_is_unlocked() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();
        let parent = f.validated(MockTransaction::eip1559().rng_hash().with_nonce(0));
        pool.add_transaction(Arc::new(parent.clone()), 0);

        // The sender has a single pending transaction when the iterator is created.
        let mut best = pool.best();
        let (tx_sender, tx_receiver) =
            tokio::sync::broadcast::channel::<PendingTransaction<MockOrdering>>(10);
        best.new_transaction_receiver = Some(tx_receiver);

        let child = f.validated(parent.transaction.clone().rng_hash().with_nonce(1));
        tx_sender
            .send(PendingTransaction {
                submission_id: 1,
                transaction: Arc::new(child.clone()),
                priority: Priority::Value(0),
            })
            .unwrap();
        assert_eq!(best.next().unwrap().id(), parent.id());
        assert_eq!(best.next().unwrap().id(), child.id());
        assert!(best.next().is_none());
    }

    #[test]
    fn replaced_snapshot_transaction_is_yielded_once() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();
        let original = f.validated(
            MockTransaction::eip1559()
                .rng_hash()
                .with_nonce(0)
                .with_priority_fee(100)
                .with_max_fee(100),
        );
        pool.add_transaction(Arc::new(original.clone()), 0);

        let mut best = pool.best();
        let (tx_sender, tx_receiver) =
            tokio::sync::broadcast::channel::<PendingTransaction<MockOrdering>>(10);
        best.new_transaction_receiver = Some(tx_receiver);

        // Before anything is yielded, the snapshot's transaction is replaced at a higher priority
        // and an unrelated transaction arrives. The original's snapshot entry is now stale.
        let replacement = f.validated(original.transaction.rng_hash());
        let other = f.validated(MockTransaction::eip1559().rng_hash().with_nonce(0));
        for (submission_id, tx, priority) in
            [(1, replacement.clone(), 300), (2, other.clone(), 200)]
        {
            tx_sender
                .send(PendingTransaction {
                    submission_id,
                    transaction: Arc::new(tx),
                    priority: Priority::Value(priority),
                })
                .unwrap();
        }

        let yielded: Vec<_> = std::iter::from_fn(|| best.next()).map(|tx| *tx.hash()).collect();
        // Tombstones only ever name transactions still in `all`, so `size_hint` cannot underflow.
        assert!(best.yielded.iter().all(|id| best.all.contains_key(id)));
        let _ = best.size_hint();
        // The stale original is skipped rather than yielded a second time.
        assert_eq!(yielded, vec![*replacement.hash(), *other.hash()]);
    }

    #[test]
    fn yielded_live_updates_do_not_accumulate() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();
        let first = f.validated(MockTransaction::eip1559().rng_hash());
        pool.add_transaction(Arc::new(first), 0);

        let mut best = pool.best();
        let (tx_sender, tx_receiver) =
            tokio::sync::broadcast::channel::<PendingTransaction<MockOrdering>>(10);
        best.new_transaction_receiver = Some(tx_receiver);
        best.allow_updates_out_of_order();
        assert!(best.next().is_some());

        for submission_id in 1..=5 {
            let tx = f.validated(MockTransaction::eip1559().rng_hash());
            tx_sender
                .send(PendingTransaction {
                    submission_id,
                    transaction: Arc::new(tx),
                    priority: Priority::Value(0),
                })
                .unwrap();
            assert!(best.next().is_some());
            // Yielded transactions are retained up to the snapshot size, one here.
            assert!(best.yielded.len() <= 1);
            assert!(best.all.len() <= 1);
        }
    }

    #[test]
    fn re_added_yielded_transaction_blocks_its_descendant() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();
        let parent = f.validated(MockTransaction::eip1559().rng_hash().with_nonce(0));
        pool.add_transaction(Arc::new(parent.clone()), 0);

        let mut best = pool.best();
        let (tx_sender, tx_receiver) =
            tokio::sync::broadcast::channel::<PendingTransaction<MockOrdering>>(10);
        best.new_transaction_receiver = Some(tx_receiver);
        assert_eq!(best.next().unwrap().id(), parent.id());

        // A replacement with the yielded id is stashed (higher priority), so it is tracked again
        // and its descendant must wait behind it rather than becoming independent.
        let replacement = f.validated(parent.transaction.clone().rng_hash());
        let child = f.validated(parent.transaction.rng_hash().with_nonce(1));
        for (submission_id, tx, priority) in [(1, replacement, 1_000), (2, child.clone(), 0)] {
            tx_sender
                .send(PendingTransaction {
                    submission_id,
                    transaction: Arc::new(tx),
                    priority: Priority::Value(priority),
                })
                .unwrap();
        }
        best.add_new_transactions();
        assert!(best.all.contains_key(child.id()));
        assert!(!best.independent.iter().any(|tx| tx.transaction.id() == child.id()));
    }

    #[test]
    fn test_best_add_transaction_with_ancestor() {
        // Initialize a new PendingPool with default MockOrdering and MockTransactionFactory
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        // Add 5 transactions with increasing nonces to the pool
        let num_tx = 5;
        let tx = MockTransaction::eip1559();
        for nonce in 0..num_tx {
            let tx = tx.clone().rng_hash().with_nonce(nonce);
            let valid_tx = f.validated(tx);
            pool.add_transaction(Arc::new(valid_tx), 0);
        }

        // Create a BestTransactions iterator from the pool
        let mut best = pool.best();

        // Use a broadcast channel for transaction updates
        let (tx_sender, tx_receiver) =
            tokio::sync::broadcast::channel::<PendingTransaction<MockOrdering>>(1000);
        best.new_transaction_receiver = Some(tx_receiver);

        // Create a new transaction with nonce 5 and validate it
        let base_tx1 = MockTransaction::eip1559().rng_hash().with_nonce(5);
        let valid_new_tx1 = f.validated(base_tx1.clone());

        // Send the new transaction through the broadcast channel
        let pending_tx1 = PendingTransaction {
            submission_id: 10,
            transaction: Arc::new(valid_new_tx1.clone()),
            priority: Priority::Value(1000),
        };
        tx_sender.send(pending_tx1.clone()).unwrap();

        // Add new transactions to the iterator
        best.add_new_transactions();

        // Verify that the new transaction has been added to the 'all' map
        assert_eq!(best.all.len(), 6);
        assert!(best.all.contains_key(valid_new_tx1.id()));

        // Verify that the new transaction has been added to the 'independent' set
        assert_eq!(best.independent.len(), 2);
        assert!(best.independent.contains(&pending_tx1));

        // Attempt to add a new transaction with a different nonce (not a duplicate)
        let base_tx2 = base_tx1.with_nonce(6);
        let valid_new_tx2 = f.validated(base_tx2);

        // Send the new transaction through the broadcast channel
        let pending_tx2 = PendingTransaction {
            submission_id: 11, // Different submission ID
            transaction: Arc::new(valid_new_tx2.clone()),
            priority: Priority::Value(1000),
        };
        tx_sender.send(pending_tx2.clone()).unwrap();

        // Add new transactions to the iterator
        best.add_new_transactions();

        // Verify that the new transaction has been added to 'all'
        assert_eq!(best.all.len(), 7);
        assert!(best.all.contains_key(valid_new_tx2.id()));

        // Verify that the new transaction has not been added to the 'independent' set
        assert_eq!(best.independent.len(), 2);
        assert!(!best.independent.contains(&pending_tx2));
    }

    #[test]
    fn test_best_transactions_filter_trait_object() {
        // Initialize a new PendingPool with default MockOrdering and MockTransactionFactory
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        // Add 5 transactions with increasing nonces to the pool
        let num_tx = 5;
        let tx = MockTransaction::eip1559();
        for nonce in 0..num_tx {
            let tx = tx.clone().rng_hash().with_nonce(nonce);
            let valid_tx = f.validated(tx);
            pool.add_transaction(Arc::new(valid_tx), 0);
        }

        // Create a trait object of BestTransactions iterator from the pool
        let best: Box<dyn crate::traits::BestTransactions<Item = _>> = Box::new(pool.best());

        // Create a filter that only returns transactions with even nonces
        let filter =
            BestTransactionFilter::new(best, |tx: &Arc<ValidPoolTransaction<MockTransaction>>| {
                tx.nonce().is_multiple_of(2)
            });

        // Verify that the filter only returns transactions with even nonces
        for tx in filter {
            assert_eq!(tx.nonce() % 2, 0);
        }
    }

    #[test]
    fn test_best_transactions_prioritized_senders() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        // Add 5 plain transactions from different senders with increasing gas price
        for gas_price in 0..5 {
            let tx = MockTransaction::eip1559().with_gas_price((gas_price + 1) * 10);
            let valid_tx = f.validated(tx);
            pool.add_transaction(Arc::new(valid_tx), 0);
        }

        // Add another transaction with 5 gas price that's going to be prioritized by sender
        let prioritized_tx = MockTransaction::eip1559().with_gas_price(5).with_gas_limit(200);
        let valid_prioritized_tx = f.validated(prioritized_tx.clone());
        pool.add_transaction(Arc::new(valid_prioritized_tx), 0);

        // Add another transaction with 3 gas price that should not be prioritized by sender because
        // of gas limit.
        let prioritized_tx2 = MockTransaction::eip1559().with_gas_price(3);
        let valid_prioritized_tx2 = f.validated(prioritized_tx2.clone());
        pool.add_transaction(Arc::new(valid_prioritized_tx2), 0);

        let prioritized_senders =
            AddressSet::from_iter([prioritized_tx.sender(), prioritized_tx2.sender()]);
        let best =
            BestTransactionsWithPrioritizedSenders::new(prioritized_senders, 200, pool.best());

        // Verify that the prioritized transaction is returned first
        // and the rest are returned in the reverse order of gas price
        let mut iter = best.into_iter();
        let top_of_block_tx = iter.next().unwrap();
        assert_eq!(top_of_block_tx.max_fee_per_gas(), 5);
        assert_eq!(top_of_block_tx.sender(), prioritized_tx.sender());
        for gas_price in (0..5).rev() {
            assert_eq!(iter.next().unwrap().max_fee_per_gas(), (gas_price + 1) * 10);
        }

        // Due to the gas limit, the transaction from second-prioritized sender was not
        // prioritized.
        let top_of_block_tx2 = iter.next().unwrap();
        assert_eq!(top_of_block_tx2.max_fee_per_gas(), 3);
        assert_eq!(top_of_block_tx2.sender(), prioritized_tx2.sender());
    }

    #[test]
    fn test_best_with_fees_iter_no_blob_fee_required() {
        // Tests transactions without blob fees where base fees are checked.
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        let base_fee: u64 = 10;
        let base_fee_per_blob_gas: u64 = 0; // No blob fee requirement

        // Insert transactions with max_fee_per_gas above the base fee
        for nonce in 0..5 {
            let tx = MockTransaction::eip1559()
                .rng_hash()
                .with_nonce(nonce)
                .with_max_fee(base_fee as u128 + 5);
            let valid_tx = f.validated(tx);
            pool.add_transaction(Arc::new(valid_tx), 0);
        }

        let mut best = pool.best_with_basefee_and_blobfee(base_fee, base_fee_per_blob_gas);

        // All transactions should be returned as no blob fee requirement is imposed
        for nonce in 0..5 {
            let tx = best.next().expect("Transaction should be returned");
            assert_eq!(tx.nonce(), nonce);
        }

        // Ensure no more transactions are left
        assert!(best.next().is_none());
    }

    #[test]
    fn test_best_with_fees_iter_mix_of_blob_and_non_blob_transactions() {
        // Tests mixed scenarios with both blob and non-blob transactions.
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        let base_fee: u64 = 10;
        let base_fee_per_blob_gas: u64 = 15;

        // Add a non-blob transaction that satisfies the base fee
        let tx_non_blob =
            MockTransaction::eip1559().rng_hash().with_nonce(0).with_max_fee(base_fee as u128 + 5);
        pool.add_transaction(Arc::new(f.validated(tx_non_blob.clone())), 0);

        // Add a blob transaction that satisfies both base fee and blob fee
        let tx_blob = MockTransaction::eip4844()
            .rng_hash()
            .with_nonce(1)
            .with_max_fee(base_fee as u128 + 5)
            .with_blob_fee(base_fee_per_blob_gas as u128 + 5);
        pool.add_transaction(Arc::new(f.validated(tx_blob.clone())), 0);

        let mut best = pool.best_with_basefee_and_blobfee(base_fee, base_fee_per_blob_gas);

        // Verify both transactions are returned
        let tx = best.next().expect("Transaction should be returned");
        assert_eq!(tx.transaction, tx_non_blob);

        let tx = best.next().expect("Transaction should be returned");
        assert_eq!(tx.transaction, tx_blob);

        // Ensure no more transactions are left
        assert!(best.next().is_none());
    }

    #[test]
    fn test_best_transactions_with_skipping_blobs() {
        // Tests the skip_blobs functionality to ensure blob transactions are skipped.
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        // Add a blob transaction
        let tx_blob = MockTransaction::eip4844().rng_hash().with_nonce(0).with_blob_fee(100);
        let valid_blob_tx = f.validated(tx_blob);
        pool.add_transaction(Arc::new(valid_blob_tx), 0);

        // Add a non-blob transaction
        let tx_non_blob = MockTransaction::eip1559().rng_hash().with_nonce(1).with_max_fee(200);
        let valid_non_blob_tx = f.validated(tx_non_blob.clone());
        pool.add_transaction(Arc::new(valid_non_blob_tx), 0);

        let mut best = pool.best();
        best.skip_blobs();

        // Only the non-blob transaction should be returned
        let tx = best.next().expect("Transaction should be returned");
        assert_eq!(tx.transaction, tx_non_blob);

        // Ensure no more transactions are left
        assert!(best.next().is_none());
    }

    #[test]
    fn test_best_transactions_no_updates() {
        // Tests the no_updates functionality to ensure it properly clears the
        // new_transaction_receiver.
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        // Add a transaction
        let tx = MockTransaction::eip1559().rng_hash().with_nonce(0).with_max_fee(100);
        let valid_tx = f.validated(tx);
        pool.add_transaction(Arc::new(valid_tx), 0);

        let mut best = pool.best();

        // Use a broadcast channel for transaction updates
        let (_tx_sender, tx_receiver) =
            tokio::sync::broadcast::channel::<PendingTransaction<MockOrdering>>(1000);
        best.new_transaction_receiver = Some(tx_receiver);

        // Ensure receiver is set
        assert!(best.new_transaction_receiver.is_some());

        // Call no_updates to clear the receiver
        best.no_updates();

        // Ensure receiver is cleared
        assert!(best.new_transaction_receiver.is_none());
    }

    #[test]
    fn test_best_transactions_yields_updates_after_empty() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut best = pool.best();
        best.allow_updates_out_of_order();

        assert!(best.next().is_none());

        let mut f = MockTransactionFactory::default();
        let tx = MockTransaction::eip1559().rng_hash();
        let valid_tx = Arc::new(f.validated(tx));
        let expected_hash = *valid_tx.hash();
        pool.add_transaction(valid_tx, 0);

        assert_eq!(*best.next().expect("new transaction should be yielded").hash(), expected_hash);
    }

    #[test]
    fn test_best_update_transaction_priority() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        // Add 5 transactions with increasing nonces to the pool
        let num_tx = 5;
        let tx = MockTransaction::eip1559();
        for nonce in 0..num_tx {
            let tx = tx.clone().rng_hash().with_nonce(nonce);
            let valid_tx = f.validated(tx);
            pool.add_transaction(Arc::new(valid_tx), 0);
        }

        // Create a BestTransactions iterator from the pool
        let mut best = pool.best();

        // Use a broadcast channel for transaction updates
        let (tx_sender, tx_receiver) =
            tokio::sync::broadcast::channel::<PendingTransaction<MockOrdering>>(1000);
        best.new_transaction_receiver = Some(tx_receiver);

        // yield one tx, effectively locking in the highest prio
        let first = best.next().unwrap();

        // Create a new transaction with nonce 5 and validate it
        let new_higher_fee_tx = MockTransaction::eip1559().with_nonce(0);
        let valid_new_higher_fee_tx = f.validated(new_higher_fee_tx);

        // Send the new transaction through the broadcast channel
        let pending_tx = PendingTransaction {
            submission_id: 10,
            transaction: Arc::new(valid_new_higher_fee_tx.clone()),
            priority: Priority::Value(u128::MAX),
        };
        tx_sender.send(pending_tx).unwrap();

        // ensure that the higher prio tx is skipped since we yielded a lower one
        for tx in best {
            assert_eq!(tx.sender_id(), first.sender_id());
            assert_ne!(tx.sender_id(), valid_new_higher_fee_tx.sender_id());
        }
    }

    /// Reproduces the "Blob Transaction Ordering, Multiple Clients" Hive scenario.
    ///
    /// Sender A contributes 5-blob transactions while sender B contributes 1-blob transactions.
    /// A single payload build should be able to fill the block with 6 blobs total (5+1).
    #[test]
    fn test_blob_transaction_ordering_multiple_clients_shape() {
        let mut pool = PendingPool::new(MockOrdering::default());
        let mut f = MockTransactionFactory::default();

        let base_fee: u64 = 10;
        let base_fee_per_blob_gas: u64 = 1;
        let max_blob_count: u64 = 6;

        let sender_a = MockTransaction::eip4844()
            .with_blob_hashes(5)
            .with_max_fee(base_fee as u128 + 20)
            .with_priority_fee(base_fee as u128 + 20)
            .with_blob_fee(120);
        for nonce in 0..5u64 {
            let tx = sender_a.clone().rng_hash().with_nonce(nonce);
            pool.add_transaction(Arc::new(f.validated(tx)), 0);
        }

        let sender_b = MockTransaction::eip4844()
            .with_blob_hashes(1)
            .with_max_fee(base_fee as u128 + 20)
            .with_priority_fee(base_fee as u128 + 20)
            .with_blob_fee(100);
        for nonce in 0..5u64 {
            let tx = sender_b.clone().rng_hash().with_nonce(nonce);
            pool.add_transaction(Arc::new(f.validated(tx)), 0);
        }

        let mut best = pool.best_with_basefee_and_blobfee(base_fee, base_fee_per_blob_gas);
        let mut block_blob_count = 0u64;
        let mut included_txs = 0u64;

        while let Some(tx) = best.next() {
            if let Some(tx_blob_count) = tx.transaction.blob_count() {
                if block_blob_count + tx_blob_count > max_blob_count {
                    crate::traits::BestTransactions::mark_invalid(
                        &mut best,
                        &tx,
                        InvalidPoolTransactionError::Eip4844(
                            Eip4844PoolTransactionError::TooManyEip4844Blobs {
                                have: block_blob_count + tx_blob_count,
                                permitted: max_blob_count,
                            },
                        ),
                    );
                    continue;
                }

                block_blob_count += tx_blob_count;
                included_txs += 1;

                if block_blob_count == max_blob_count {
                    best.skip_blobs();
                    break;
                }
            }
        }

        assert_eq!(
            block_blob_count, max_blob_count,
            "expected a full blob block (5+1 blobs across senders)"
        );
        assert_eq!(included_txs, 2, "expected one 5-blob tx and one 1-blob tx in the block");
    }
}
