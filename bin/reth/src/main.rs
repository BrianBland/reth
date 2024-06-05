#![allow(missing_docs)]

// We use jemalloc for performance reasons.
#[cfg(all(feature = "jemalloc", unix))]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(all(feature = "optimism", not(test)))]
compile_error!(
    "Cannot build the `reth` binary with the `optimism` feature flag enabled. Did you
mean to build `op-reth`?"
);

#[cfg(not(feature = "optimism"))]
fn main() {
    use reth::cli::Cli;
    use reth_node_ethereum::EthereumNode;

    use reth_auto_seal_consensus::AutoSealConsensus;
    use reth_node_builder::{components::ConsensusBuilder, node::FullNodeTypes, BuilderContext};

    #[derive(Debug, Clone, Copy)]
    struct DevConsensusBuilder;

    impl<Node> ConsensusBuilder<Node> for DevConsensusBuilder
    where
        Node: FullNodeTypes,
    {
        type Consensus = AutoSealConsensus;

        async fn build_consensus(
            self,
            ctx: &BuilderContext<Node>,
        ) -> eyre::Result<Self::Consensus> {
            Ok(AutoSealConsensus::new(ctx.chain_spec()))
        }
    }

    reth::sigsegv_handler::install();

    // Enable backtraces unless a RUST_BACKTRACE value has already been explicitly provided.
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }

    if let Err(err) = Cli::parse_args().run(|builder, _| async {
        let is_dev_mode = builder.config().dev.dev;
        if is_dev_mode {
            let handle = builder
                .with_types::<EthereumNode>()
                .with_components(EthereumNode::components().consensus(DevConsensusBuilder))
                .launch()
                .await?;
            handle.node_exit_future.await
        } else {
            let handle = builder.launch_node(EthereumNode::default()).await?;
            handle.node_exit_future.await
        }
    }) {
        eprintln!("Error: {err:?}");
        std::process::exit(1);
    }
}
