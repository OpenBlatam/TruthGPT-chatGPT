use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn main() {
    let q_values = vec![1.0, 2.0, 3.0, 4.0];
    let temperature = 1.0;

    let action = policy_with_temperature(&q_values, temperature);

    println!("Selected action: {}", action);
}

fn policy_with_temperature(q_values: &[f64], temperature: f64) -> usize {
    const CACHE_SIZE: usize = 256;
    let mut cdf = 0.0;
    let mut rng = SmallRng::from_entropy();
    let rand_val = rng.gen_range(0.0, 1.0);

    let mut prob_cdf = [0.0; CACHE_SIZE + 1];
    for i in (0..q_values.len()).step_by(CACHE_SIZE) {
        let cache_size = q_values.len() - i;
        let cache_size = if cache_size >= CACHE_SIZE { CACHE_SIZE } else { cache_size };

        let scaled_q_vec = &q_values[i..i+cache_size]
            .iter()
            .map(|&x| x / temperature)
            .collect::<Vec<f64>>();
        let total: f64 = scaled_q_vec
            .iter()
            .map(|&x| x.exp())
            .sum();
        for (j, &scaled_q) in scaled_q_vec.iter().enumerate() {
            let exp = scaled_q.exp();
            let prob = exp / total;
            cdf += prob;
            prob_cdf[i/CACHE_SIZE+j+1] = cdf;
        }
    }

    let action = prob_cdf.iter()
        .enumerate()
        .find(|(_, &cdf)| cdf >= rand_val)
        .map(|(i, _)| i)
        .unwrap_or(q_values.len());

    action
}
