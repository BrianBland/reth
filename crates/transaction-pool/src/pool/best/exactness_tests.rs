//! Randomized differential tests pinning the exact behavior of [`BestTransactions`].
//!
//! `ReferenceBestTransactions` is a verbatim copy of the iterator before popped transactions became
//! tombstones; the new iterator must match it pop for pop.
//! [`crate::traits::BestTransactions::next_skipping`] must match a caller that receives each
//! claimed transaction from `next` and asks again.

use super::*;
use crate::{
    test_utils::{MockOrdering, MockTransaction, MockTransactionFactory},
    traits::BestTransactions as _,
};
use alloy_primitives::{Address, TxHash};
use std::collections::HashSet;
use tokio::sync::broadcast::{self, Sender};

type Pending = PendingTransaction<MockOrdering>;
type Tx = Arc<ValidPoolTransaction<MockTransaction>>;

/// The iterator as of the fork point, kept verbatim apart from its name.
struct ReferenceBestTransactions {
    all: OrdMap<TransactionId, Pending>,
    independent: BTreeSet<Pending>,
    invalid: FxHashSet<SenderId>,
    new_transaction_receiver: Option<Receiver<Pending>>,
    last_priority: Option<Priority<u128>>,
    skip_blobs: bool,
    allow_updates_out_of_order: bool,
}

impl ReferenceBestTransactions {
    fn mark_invalid(&mut self, tx: &Tx) {
        self.invalid.insert(tx.sender_id());
    }

    fn ancestor(&self, id: &TransactionId) -> Option<&Pending> {
        self.all.get(&id.unchecked_ancestor()?)
    }

    fn try_recv(&mut self) -> Option<IncomingTransaction<MockOrdering>> {
        loop {
            match self.new_transaction_receiver.as_mut()?.try_recv() {
                Ok(tx) => {
                    if !self.allow_updates_out_of_order &&
                        let Some(last_priority) = &self.last_priority &&
                        &tx.priority > last_priority
                    {
                        return Some(IncomingTransaction::Stash(tx))
                    }
                    return Some(IncomingTransaction::Process(tx))
                }
                Err(TryRecvError::Lagged(_)) => {}
                Err(_) => return None,
            }
        }
    }

    fn pop_best(&mut self) -> Option<Pending> {
        self.independent.pop_last().inspect(|best| {
            self.all.remove(best.transaction.id());
        })
    }

    fn add_new_transactions(&mut self) {
        for _ in 0..MAX_NEW_TRANSACTIONS_PER_BATCH {
            if let Some(pending_tx) = self.try_recv() {
                match pending_tx {
                    IncomingTransaction::Process(tx) => {
                        let tx_id = *tx.transaction.id();
                        if self.ancestor(&tx_id).is_none() {
                            self.independent.insert(tx.clone());
                        }
                        self.all.insert(tx_id, tx);
                    }
                    IncomingTransaction::Stash(tx) => {
                        let tx_id = *tx.transaction.id();
                        self.all.insert(tx_id, tx);
                    }
                }
            } else {
                break;
            }
        }
    }

    fn next_tx_and_priority(&mut self) -> Option<(Tx, Priority<u128>)> {
        loop {
            self.add_new_transactions();
            let best = self.pop_best()?;
            let sender_id = best.transaction.sender_id();
            if self.invalid.contains(&sender_id) {
                continue
            }
            if let Some(unlocked) = self.all.get(&best.unlocks()) {
                self.independent.insert(unlocked.clone());
            }
            if self.skip_blobs && best.transaction.is_eip4844() {
                self.mark_invalid(&best.transaction)
            } else {
                if self.new_transaction_receiver.is_some() {
                    self.last_priority = Some(best.priority.clone())
                }
                return Some((best.transaction, best.priority))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.new_transaction_receiver.is_none().then_some(self.all.len()))
    }
}

/// A deterministic xorshift generator, so failures reproduce from their seed.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

/// Transactions of a randomized script: a snapshot of gapless sender chains and the live updates
/// a script may send later.
struct Universe {
    factory: MockTransactionFactory,
    senders: Vec<Address>,
    /// Next unused nonce per sender, so arrivals can extend a chain.
    next_nonce: Vec<u64>,
    next_submission_id: u64,
    /// Every pending transaction created so far, so arrivals can replace or repeat one.
    created: Vec<Pending>,
}

impl Universe {
    /// Priorities come from a narrow range so that exact ties, broken by submission id, are
    /// common.
    fn transaction(&mut self, rng: &mut Rng, sender: usize, nonce: u64) -> Pending {
        let priority = 1 + rng.below(8) as u128;
        let base = if rng.chance(10) {
            MockTransaction::eip4844().with_blob_fee(1 + rng.below(3) as u128)
        } else {
            MockTransaction::eip1559()
        };
        let max_fee = priority + rng.below(6) as u128;
        let tx = base
            .with_sender(self.senders[sender])
            .with_nonce(nonce)
            .with_priority_fee(priority)
            .with_max_fee(max_fee)
            .rng_hash();
        let transaction = Arc::new(self.factory.validated(tx));
        let submission_id = self.next_submission_id;
        self.next_submission_id += 1;
        let pending =
            PendingTransaction { submission_id, transaction, priority: Priority::Value(priority) };
        self.created.push(pending.clone());
        pending
    }

    fn snapshot(rng: &mut Rng) -> (Self, BestTransactions<MockOrdering>) {
        let sender_count = 1 + rng.below(6) as usize;
        let mut universe = Self {
            factory: MockTransactionFactory::default(),
            senders: (0..sender_count).map(|_| Address::random()).collect(),
            next_nonce: vec![0; sender_count],
            next_submission_id: 0,
            created: Vec::new(),
        };
        let mut all = OrdMap::new();
        let mut independent = Vec::new();
        for sender in 0..sender_count {
            let first_nonce = rng.below(3);
            let chain = rng.below(5);
            for nonce in first_nonce..first_nonce + chain {
                let pending = universe.transaction(rng, sender, nonce);
                if nonce == first_nonce {
                    independent.push(pending.clone());
                }
                all.insert(*pending.transaction.id(), pending);
            }
            universe.next_nonce[sender] = first_nonce + chain;
        }
        let best = BestTransactions {
            all,
            popped: Default::default(),
            independent: independent.into_iter().collect(),
            invalid: Default::default(),
            new_transaction_receiver: None,
            last_priority: None,
            skip_blobs: false,
            allow_updates_out_of_order: false,
        };
        (universe, best)
    }

    /// A live update: a new chain head, the next nonce of a chain, a replacement of any
    /// transaction created so far (possibly already popped), or a repeat of one.
    fn arrival(&mut self, rng: &mut Rng) -> Pending {
        match rng.below(10) {
            0..=3 => {
                let sender = rng.below(self.senders.len() as u64) as usize;
                let nonce = self.next_nonce[sender];
                self.next_nonce[sender] += 1;
                self.transaction(rng, sender, nonce)
            }
            4..=7 if !self.created.is_empty() => {
                let replaced = self.created[rng.below(self.created.len() as u64) as usize].clone();
                let sender =
                    self.senders.iter().position(|s| *s == replaced.transaction.sender()).unwrap();
                self.transaction(rng, sender, replaced.transaction.nonce())
            }
            8 if !self.created.is_empty() => {
                self.created[rng.below(self.created.len() as u64) as usize].clone()
            }
            _ => {
                let sender = self.senders.len();
                self.senders.push(Address::random());
                self.next_nonce.push(1);
                self.transaction(rng, sender, 0)
            }
        }
    }
}

fn reference_of(best: &BestTransactions<MockOrdering>) -> ReferenceBestTransactions {
    ReferenceBestTransactions {
        all: best.all.clone(),
        independent: best.independent.iter().cloned().collect(),
        invalid: Default::default(),
        new_transaction_receiver: None,
        last_priority: None,
        skip_blobs: false,
        allow_updates_out_of_order: false,
    }
}

/// Sends `count` live updates through `sender`.
fn send_arrivals(universe: &mut Universe, rng: &mut Rng, sender: &Sender<Pending>, count: u64) {
    for _ in 0..count {
        let _ = sender.send(universe.arrival(rng));
    }
}

fn identity(item: Option<(Tx, Priority<u128>)>) -> Option<(TxHash, u64)> {
    item.map(|(tx, _)| (*tx.hash(), tx.nonce()))
}

/// The tombstone iterator pops exactly what the fork-point iterator pops under
/// random scripts of pops, invalidations, blob skipping and live updates, including replacements
/// of popped and unpopped ids, repeats and batches beyond one update round.
#[test]
fn matches_reference_iterator() {
    for seed in 0..2_000 {
        let mut rng = Rng::new(seed);
        let (mut universe, mut best) = Universe::snapshot(&mut rng);
        let mut reference = reference_of(&best);
        let (sender, receiver) = broadcast::channel(1_000);
        if rng.chance(80) {
            best.new_transaction_receiver = Some(receiver);
            reference.new_transaction_receiver = Some(sender.subscribe());
        }
        if rng.chance(30) {
            best.allow_updates_out_of_order = true;
            reference.allow_updates_out_of_order = true;
        }
        let mut yielded = Vec::new();
        for step in 0..60 {
            match rng.below(20) {
                0..=10 => {
                    let expected = reference.next_tx_and_priority();
                    let actual = best.next_tx_and_priority();
                    assert_eq!(identity(actual), identity(expected.clone()), "seed {seed} {step}");
                    if let Some((tx, _)) = expected {
                        yielded.push(tx);
                    }
                }
                11..=12 if !yielded.is_empty() => {
                    let tx = &yielded[rng.below(yielded.len() as u64) as usize];
                    reference.mark_invalid(tx);
                    best.mark_invalid(tx, InvalidPoolTransactionError::Underpriced);
                }
                13..=16 => {
                    let count = 1 + rng.below(24);
                    send_arrivals(&mut universe, &mut rng, &sender, count);
                }
                17 => {
                    let skip = rng.chance(50);
                    reference.skip_blobs = skip;
                    best.set_skip_blobs(skip);
                }
                18 if rng.chance(20) => {
                    reference.new_transaction_receiver.take();
                    reference.last_priority.take();
                    best.no_updates();
                }
                _ => {}
            }
            assert_eq!(best.size_hint(), reference.size_hint(), "seed {seed} step {step}");
        }
    }
}

/// Drives a natural iterator and a skipping one over the same script: the skipping iterator
/// must return exactly what the natural one returns once every claimed transaction is dropped.
/// Returns how many transactions were claimed, and passes every candidate `skip` sees to `seen`
/// together with whether blobs are skipped.
fn assert_skipping_matches<I>(
    seed: u64,
    rng: &mut Rng,
    universe: &mut Universe,
    sender: &Sender<Pending>,
    natural: &mut I,
    skipping: &mut I,
    mut seen: impl FnMut(&Tx, bool),
) -> usize
where
    I: crate::traits::BestTransactions<Item = Tx>,
{
    let claim_bucket = rng.below(4);
    let claims = |tx: &Tx| tx.hash()[0] as u64 % 4 == claim_bucket;
    let mut claimed = 0;
    let mut skip_blobs = false;
    let mut yielded = Vec::new();
    for step in 0..60 {
        match rng.below(20) {
            0..=11 => {
                let actual = skipping.next_skipping(&mut |tx| {
                    seen(tx, skip_blobs);
                    claims(tx)
                });
                let expected = loop {
                    match natural.next() {
                        Some(tx) if claims(&tx) => claimed += 1,
                        other => break other,
                    }
                };
                assert_eq!(
                    actual.as_ref().map(|tx| *tx.hash()),
                    expected.as_ref().map(|tx| *tx.hash()),
                    "seed {seed} step {step}"
                );
                yielded.extend(expected);
            }
            12..=13 if !yielded.is_empty() => {
                let tx = &yielded[rng.below(yielded.len() as u64) as usize];
                natural.mark_invalid(tx, InvalidPoolTransactionError::Underpriced);
                skipping.mark_invalid(tx, InvalidPoolTransactionError::Underpriced);
            }
            14..=17 => {
                let count = 1 + rng.below(24);
                send_arrivals(universe, rng, sender, count);
            }
            18 => {
                skip_blobs = rng.chance(50);
                natural.set_skip_blobs(skip_blobs);
                skipping.set_skip_blobs(skip_blobs);
            }
            _ => {}
        }
    }
    claimed
}

/// Claiming a transaction through `next_skipping` equals receiving it from `next` and asking
/// again: the same unlocks, live-update batches and priority bound for stashing updates.
#[test]
fn next_skipping_matches_next_without_claimed() {
    let mut claimed = 0;
    for seed in 0..2_000 {
        let mut rng = Rng::new(seed);
        let (mut universe, mut natural) = Universe::snapshot(&mut rng);
        let mut skipping = BestTransactions {
            all: natural.all.clone(),
            popped: Default::default(),
            independent: natural.independent.iter().cloned().collect(),
            invalid: Default::default(),
            new_transaction_receiver: None,
            last_priority: None,
            skip_blobs: false,
            allow_updates_out_of_order: false,
        };
        let (sender, receiver) = broadcast::channel(1_000);
        if rng.chance(80) {
            natural.new_transaction_receiver = Some(receiver);
            skipping.new_transaction_receiver = Some(sender.subscribe());
        }
        claimed += assert_skipping_matches(
            seed,
            &mut rng,
            &mut universe,
            &sender,
            &mut natural,
            &mut skipping,
            |_, _| {},
        );
    }
    assert!(claimed > 1_000, "scripts must claim transactions: {claimed}");
}

/// Under fees, `next_skipping` still matches `next`, and `skip` never sees a transaction the fee
/// check or blob skipping would drop: those keep their invalidation path.
#[test]
fn next_skipping_with_fees_only_offers_payable_transactions() {
    let mut offered_blob_fee_checked = false;
    for seed in 0..2_000 {
        let mut rng = Rng::new(seed);
        let (mut universe, natural) = Universe::snapshot(&mut rng);
        let (base_fee, blob_fee) = (rng.below(6), rng.below(3));
        let skipping = BestTransactions {
            all: natural.all.clone(),
            popped: Default::default(),
            independent: natural.independent.iter().cloned().collect(),
            invalid: Default::default(),
            new_transaction_receiver: None,
            last_priority: None,
            skip_blobs: false,
            allow_updates_out_of_order: false,
        };
        let mut natural =
            BestTransactionsWithFees { best: natural, base_fee, base_fee_per_blob_gas: blob_fee };
        let mut skipping =
            BestTransactionsWithFees { best: skipping, base_fee, base_fee_per_blob_gas: blob_fee };
        let (sender, receiver) = broadcast::channel(1_000);
        natural.best.new_transaction_receiver = Some(receiver);
        skipping.best.new_transaction_receiver = Some(sender.subscribe());
        let mut offered = HashSet::new();
        assert_skipping_matches(
            seed,
            &mut rng,
            &mut universe,
            &sender,
            &mut natural,
            &mut skipping,
            |tx, skip_blobs| {
                assert!(satisfies_fees(tx, base_fee, blob_fee), "seed {seed}: underpriced offer");
                assert!(!(skip_blobs && tx.is_eip4844()), "seed {seed}: skipped blob offer");
                offered.insert(*tx.hash());
            },
        );
        offered_blob_fee_checked |= blob_fee > 0 && !offered.is_empty();
    }
    assert!(offered_blob_fee_checked);
}
