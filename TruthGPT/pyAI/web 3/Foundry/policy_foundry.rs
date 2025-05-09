use foundry::prelude::*;

fn policy_with_temperature(q_values: &[f64], temperature: f64) -> usize {
    let scaled_q_values = q_values
        .iter()
        .map(|&x| x.exp() / temperature)
        .collect::<Vec<_>>();

    let total = scaled_q_values.iter().sum::<f64>();
    let probabilities = scaled_q_values.iter().map(|&x| x / total).collect::<Vec<_>>();

    let mut rng = Foundry::with_seed(42);
    let rand_val = rng.gen_range(0.0, 1.0);

    let cdf = probabilities
        .iter()
        .scan(0.0, |state, &x| {
            *state += x;
            Some(*state)
        })
        .collect::<Vec<_>>();

    let mut action = 0;
    for (i, &x) in cdf.iter().enumerate() {
        if x >= rand_val {
            action = i;
            break;
        }
    }

    action
}
