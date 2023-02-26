use futures::stream::{self, StreamExt};
use tokio::runtime::Runtime;
use foundry::prelude::*;

async fn async_policy_with_temperature(q_values: &[f64], temperature: f64) -> usize {
    // Compute the scaled Q-values and total
    let scaled_q_values = q_values.iter().map(|&x| x.exp() / temperature).collect::<Vec<_>>();
    let total: f64 = scaled_q_values.iter().sum();

    // Compute the probabilities by normalizing the scaled Q-values
    let probabilities = scaled_q_values.iter().map(|&x| x / total).collect::<Vec<_>>();

    // Generate a random number between 0 and 1
    let mut rng = Foundry::with_seed(42);
    let rand_val = rng.gen_range(0.0, 1.0);

    // Compute the cumulative distribution function (CDF) of the probabilities
    let cdf = stream::iter(probabilities.iter())
        .scan(0.0, |state, &x| {
            *state += x;
            Some(*state)
        })
        .collect::<Vec<_>>()
        .await;

    // Find the first index where the CDF is greater than or equal to the random value
    cdf.iter().enumerate().find(|&(_, &x)| x >= rand_val).map_or(q_values.len(), |(i, _)| i)
}

fn policy_with_temperature(q_values: &[f64], temperature: f64) -> usize {
    let mut rt = Runtime::new().unwrap();
    rt.block_on(async_policy_with_temperature(q_values, temperature))
}
