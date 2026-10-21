#![allow(missing_docs)]

//! Benchmarks for the `best_transactions` iterator a block builder drains: creating it from a
//! large pending pool, and yielding the first few hundred, few thousand, or all transactions,
//! with and without new transactions arriving while it is iterated.

use alloy_primitives::{Address, B256, U256};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use reth_primitives_traits::SealedBlock;
use reth_transaction_pool::{
    blobstore::InMemoryBlobStore,
    noop::MockTransactionValidator,
    pool::PoolInner,
    test_utils::{MockOrdering, MockTransaction},
    validate::ValidTransaction,
    CanonicalStateUpdate, PoolConfig, PoolTransaction, PoolUpdateKind, SubPoolLimit,
    TransactionOrigin, TransactionValidationOutcome,
};

type BenchPool =
    PoolInner<MockTransactionValidator<MockTransaction>, MockOrdering, InMemoryBlobStore>;

const BASE_FEE: u64 = 100;

/// Transactions per sender, cycled over senders: mostly single transactions, some short nonce
/// chains.
const CHAIN_LENGTHS: [u64; 6] = [1, 1, 1, 1, 2, 4];

/// Transactions that arrive while the iterator is drained in the `live` cases: half from new
/// senders, half the next nonce of a sender already in the pool.
const ARRIVALS: usize = 200;

fn sender_address(index: usize) -> Address {
    let mut bytes = [0u8; 20];
    bytes[12..].copy_from_slice(&(index as u64 + 1).to_be_bytes());
    Address::from(bytes)
}

/// A transaction whose tip is spread over a thousand levels by `seed`.
fn make_tx(sender: usize, nonce: u64, seed: usize) -> MockTransaction {
    let tip = 1 + (seed.wrapping_mul(7919) % 1_000) as u128;
    let mut tx = MockTransaction::eip1559();
    tx.set_sender(sender_address(sender))
        .set_nonce(nonce)
        .set_max_fee(BASE_FEE as u128 + tip)
        .set_priority_fee(tip)
        .set_gas_limit(21_000);
    tx
}

/// `tx` validated against a sender nonce of zero.
const fn outcome(
    tx: MockTransaction,
) -> (TransactionOrigin, TransactionValidationOutcome<MockTransaction>) {
    let outcome = TransactionValidationOutcome::Valid {
        balance: U256::MAX,
        state_nonce: 0,
        bytecode_hash: None,
        transaction: ValidTransaction::Valid(tx),
        propagate: false,
        authorities: None,
    };
    (TransactionOrigin::External, outcome)
}

/// An uncapped pool with `size` pending transactions, and the next free sender index.
fn build_pool(size: usize) -> (BenchPool, usize) {
    let pool = PoolInner::new(
        MockTransactionValidator::default(),
        MockOrdering::default(),
        InMemoryBlobStore::default(),
        PoolConfig { pending_limit: SubPoolLimit::max(), ..Default::default() },
    );
    let mut block = reth_ethereum_primitives::Block::default();
    block.header.gas_limit = 30_000_000;
    block.header.base_fee_per_gas = Some(BASE_FEE);
    pool.on_canonical_state_change(CanonicalStateUpdate {
        new_tip: &SealedBlock::seal_slow(block),
        pending_block_base_fee: BASE_FEE,
        pending_block_blob_fee: None,
        changed_accounts: Vec::new(),
        mined_transactions: Vec::new(),
        update_kind: PoolUpdateKind::Commit,
    });

    let mut batch = Vec::with_capacity(size);
    let mut sender = 0;
    while batch.len() < size {
        let length = CHAIN_LENGTHS[sender % CHAIN_LENGTHS.len()].min((size - batch.len()) as u64);
        for nonce in 0..length {
            batch.push(outcome(make_tx(sender, nonce, batch.len())));
        }
        sender += 1;
    }
    pool.add_transactions_with_origins(batch);
    assert_eq!(pool.size().pending, size, "every transaction is pending");
    (pool, sender)
}

fn bench_best_transactions(c: &mut Criterion) {
    let mut group = c.benchmark_group("best_transactions");
    group.sample_size(30);

    for size in [5_000, 20_000] {
        let (pool, senders) = build_pool(size);

        group.bench_function(BenchmarkId::new("init", size), |b| {
            b.iter_batched(|| (), |()| pool.best_transactions(), BatchSize::SmallInput)
        });

        for take in [200, 1_000, size] {
            let name = if take == size { "all".to_string() } else { format!("take_{take}") };
            group.bench_function(BenchmarkId::new(name, size), |b| {
                b.iter_batched(
                    || (),
                    |()| {
                        let mut best = pool.best_transactions();
                        let yielded = best.by_ref().take(take).count();
                        assert_eq!(yielded, take);
                        best
                    },
                    BatchSize::SmallInput,
                )
            });
        }

        // The iterator is created, then `ARRIVALS` transactions are added to the pool before it is
        // drained, so it sees them as live updates. Each setup first removes the previous
        // iteration's arrivals, outside the timing.
        let arrivals: Vec<_> = (0..ARRIVALS)
            .map(|i| {
                if i % 2 == 0 {
                    make_tx(senders + i, 0, size + i)
                } else {
                    // Every sender index `s` with `s % 6 < 4` has a single transaction; append its
                    // nonce 1.
                    make_tx((i / 2) * CHAIN_LENGTHS.len(), 1, size + i)
                }
            })
            .collect();
        let hashes: Vec<B256> = arrivals.iter().map(|tx| *tx.hash()).collect();
        group.bench_function(BenchmarkId::new("live_take_1000", size), |b| {
            b.iter_batched(
                || {
                    pool.remove_transactions(hashes.clone());
                    let best = pool.best_transactions();
                    let added =
                        pool.add_transactions_with_origins(arrivals.iter().cloned().map(outcome));
                    assert!(added.iter().all(Result::is_ok), "every arrival is added");
                    best
                },
                |mut best| {
                    let yielded = best.by_ref().take(1_000).count();
                    assert_eq!(yielded, 1_000);
                    best
                },
                BatchSize::PerIteration,
            );
        });
        pool.remove_transactions(hashes.clone());
        assert_eq!(pool.size().pending, size);
    }

    group.finish();
}

criterion_group!(benches, bench_best_transactions);
criterion_main!(benches);
