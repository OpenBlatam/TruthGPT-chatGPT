/**
 * 🔥 K6 Load Testing Script
 * Comprehensive load testing for inference API
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const inferenceLatency = new Trend('inference_latency_ms');
const cacheHitRate = new Rate('cache_hits');

// Configuration
export const options = {
  stages: [
    { duration: '2m', target: 50 },   // Ramp up to 50 users
    { duration: '5m', target: 50 },   // Stay at 50 users
    { duration: '2m', target: 100 },  // Ramp up to 100 users
    { duration: '5m', target: 100 },  // Stay at 100 users
    { duration: '2m', target: 0 },    // Ramp down to 0 users
  ],
  thresholds: {
    'http_req_duration': ['p(95)<300', 'p(99)<500'], // 95% under 300ms, 99% under 500ms
    'http_req_failed': ['rate<0.01'],                 // Less than 1% errors
    'errors': ['rate<0.01'],
    'inference_latency_ms': ['p(95)<300', 'p(99)<500'],
  },
};

const API_URL = __ENV.API_URL || 'http://localhost:8080';
const API_TOKEN = __ENV.API_TOKEN || 'changeme';

const prompts = [
  'Explain quantum computing in simple terms',
  'Write a haiku about artificial intelligence',
  'What are the benefits of renewable energy?',
  'Describe the process of photosynthesis',
  'Tell me about the history of computers',
];

export default function () {
  // Select random prompt
  const prompt = prompts[Math.floor(Math.random() * prompts.length)];
  
  const payload = JSON.stringify({
    model: 'gpt-4o',
    prompt: prompt,
    params: {
      max_new_tokens: 128,
      temperature: 0.7,
    },
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${API_TOKEN}`,
    },
    tags: { name: 'InferenceAPI' },
  };

  const startTime = Date.now();
  const res = http.post(`${API_URL}/v1/infer`, payload, params);
  const latency = Date.now() - startTime;

  // Check response
  const success = check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
    'has output': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.output && body.output.length > 0;
      } catch {
        return false;
      }
    },
  });

  // Record metrics
  errorRate.add(!success);
  inferenceLatency.add(latency);
  
  // Check cache hit
  if (res.status === 200) {
    try {
      const body = JSON.parse(res.body);
      cacheHitRate.add(body.cached === true);
    } catch {}
  }

  sleep(Math.random() * 2 + 1); // Random sleep between 1-3 seconds
}

export function handleSummary(data) {
  return {
    'summary.json': JSON.stringify(data),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data, options) {
  const indent = options.indent || '  ';
  let summary = '\n📊 Load Test Summary\n';
  summary += '='.repeat(50) + '\n\n';
  
  // HTTP metrics
  summary += `${indent}HTTP Requests:\n`;
  summary += `${indent}  Total: ${data.metrics.http_reqs.values.count}\n`;
  summary += `${indent}  Rate: ${data.metrics.http_reqs.values.rate.toFixed(2)} req/s\n`;
  summary += `${indent}  Failed: ${data.metrics.http_req_failed.values.rate * 100}%\n\n`;
  
  // Duration
  summary += `${indent}Response Times:\n`;
  summary += `${indent}  p50: ${data.metrics.http_req_duration.values.med}ms\n`;
  summary += `${indent}  p95: ${data.metrics.http_req_duration.values['p(95)']}ms\n`;
  summary += `${indent}  p99: ${data.metrics.http_req_duration.values['p(99)']}ms\n\n`;
  
  // Inference latency
  if (data.metrics.inference_latency_ms) {
    summary += `${indent}Inference Latency:\n`;
    summary += `${indent}  p50: ${data.metrics.inference_latency_ms.values.med}ms\n`;
    summary += `${indent}  p95: ${data.metrics.inference_latency_ms.values['p(95)']}ms\n`;
    summary += `${indent}  p99: ${data.metrics.inference_latency_ms.values['p(99)']}ms\n\n`;
  }
  
  // Cache hit rate
  if (data.metrics.cache_hits) {
    summary += `${indent}Cache Hit Rate: ${(data.metrics.cache_hits.values.rate * 100).toFixed(2)}%\n\n`;
  }
  
  // Errors
  summary += `${indent}Error Rate: ${(data.metrics.errors.values.rate * 100).toFixed(2)}%\n`;
  
  return summary;
}


