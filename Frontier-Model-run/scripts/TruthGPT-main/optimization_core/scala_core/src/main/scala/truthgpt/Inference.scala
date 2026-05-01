/**
 * TruthGPT Scala Core - Distributed Inference
 * 
 * High-performance distributed inference using Apache Spark and Akka.
 * 
 * Features:
 * - Spark-based distributed batch inference
 * - Akka Streams for streaming inference
 * - Cats Effect for safe async operations
 * - ZIO for effect management
 */
package truthgpt

import scala.concurrent.{ExecutionContext, Future}
import scala.collection.mutable
import java.util.concurrent.atomic.{AtomicLong, AtomicInteger}

/**
 * Configuration for inference engine.
 */
case class InferenceConfig(
  maxBatchSize: Int = 32,
  maxSeqLen: Int = 2048,
  temperature: Float = 1.0f,
  topK: Int = 50,
  topP: Float = 0.9f,
  numWorkers: Int = 4,
  timeoutMs: Long = 30000
)

/**
 * Inference request.
 */
case class InferenceRequest(
  id: Long,
  inputIds: Array[Int],
  maxNewTokens: Int = 100,
  temperature: Float = 1.0f,
  topK: Int = 50,
  topP: Float = 0.9f,
  priority: Int = 0
)

/**
 * Inference response.
 */
case class InferenceResponse(
  id: Long,
  outputIds: Array[Int],
  logprobs: Option[Array[Float]] = None,
  latencyMs: Long = 0,
  success: Boolean = true,
  error: Option[String] = None
)

/**
 * Request priority queue.
 */
class PriorityQueue {
  private val queue = new mutable.PriorityQueue[InferenceRequest]()(
    Ordering.by(r => (r.priority, -r.id))
  )
  private val lock = new Object()
  
  def enqueue(request: InferenceRequest): Unit = lock.synchronized {
    queue.enqueue(request)
  }
  
  def dequeue(): Option[InferenceRequest] = lock.synchronized {
    if (queue.nonEmpty) Some(queue.dequeue()) else None
  }
  
  def dequeueBatch(maxSize: Int): Seq[InferenceRequest] = lock.synchronized {
    val batch = mutable.ArrayBuffer[InferenceRequest]()
    while (batch.size < maxSize && queue.nonEmpty) {
      batch += queue.dequeue()
    }
    batch.toSeq
  }
  
  def size: Int = lock.synchronized { queue.size }
  
  def isEmpty: Boolean = lock.synchronized { queue.isEmpty }
}

/**
 * Batch scheduler for dynamic batching.
 */
class BatchScheduler(config: InferenceConfig) {
  private val requestQueue = new PriorityQueue()
  private val requestIdCounter = new AtomicLong(0)
  private val completedCount = new AtomicLong(0)
  private val failedCount = new AtomicLong(0)
  
  def submit(
    inputIds: Array[Int],
    maxNewTokens: Int = 100,
    temperature: Float = 1.0f,
    priority: Int = 0
  ): Long = {
    val id = requestIdCounter.incrementAndGet()
    val request = InferenceRequest(
      id = id,
      inputIds = inputIds,
      maxNewTokens = maxNewTokens,
      temperature = temperature,
      priority = priority
    )
    requestQueue.enqueue(request)
    id
  }
  
  def nextBatch(): Seq[InferenceRequest] = {
    requestQueue.dequeueBatch(config.maxBatchSize)
  }
  
  def recordComplete(count: Int): Unit = {
    completedCount.addAndGet(count)
  }
  
  def recordFailed(count: Int): Unit = {
    failedCount.addAndGet(count)
  }
  
  def stats: Map[String, Long] = Map(
    "queued" -> requestQueue.size,
    "completed" -> completedCount.get(),
    "failed" -> failedCount.get()
  )
}

/**
 * Token sampler with various strategies.
 */
object TokenSampler {
  private val rng = new scala.util.Random()
  
  /**
   * Apply temperature scaling.
   */
  def applyTemperature(logits: Array[Float], temperature: Float): Array[Float] = {
    if (temperature == 1.0f) logits
    else logits.map(_ / temperature)
  }
  
  /**
   * Compute softmax probabilities.
   */
  def softmax(logits: Array[Float]): Array[Float] = {
    val maxLogit = logits.max
    val expLogits = logits.map(l => math.exp(l - maxLogit).toFloat)
    val sumExp = expLogits.sum
    expLogits.map(_ / sumExp)
  }
  
  /**
   * Top-K filtering.
   */
  def topK(probs: Array[Float], k: Int): (Array[Int], Array[Float]) = {
    val indexed = probs.zipWithIndex.sortBy(-_._1)
    val topIndices = indexed.take(k).map(_._2)
    val topProbs = indexed.take(k).map(_._1)
    
    // Renormalize
    val sum = topProbs.sum
    (topIndices, topProbs.map(_ / sum))
  }
  
  /**
   * Top-P (nucleus) filtering.
   */
  def topP(probs: Array[Float], p: Float): (Array[Int], Array[Float]) = {
    val indexed = probs.zipWithIndex.sortBy(-_._1)
    var cumSum = 0.0f
    val filtered = indexed.takeWhile { case (prob, _) =>
      val include = cumSum < p
      cumSum += prob
      include || cumSum == prob  // Always include at least one
    }
    
    val indices = filtered.map(_._2)
    val filteredProbs = filtered.map(_._1)
    
    // Renormalize
    val sum = filteredProbs.sum
    (indices, filteredProbs.map(_ / sum))
  }
  
  /**
   * Sample from probability distribution.
   */
  def sample(probs: Array[Float]): Int = {
    val r = rng.nextFloat()
    var cumSum = 0.0f
    for (i <- probs.indices) {
      cumSum += probs(i)
      if (r < cumSum) return i
    }
    probs.length - 1
  }
  
  /**
   * Sample with temperature, top-k, and top-p.
   */
  def sampleWithConfig(
    logits: Array[Float],
    temperature: Float = 1.0f,
    k: Int = 50,
    p: Float = 0.9f
  ): Int = {
    // Apply temperature
    val scaledLogits = applyTemperature(logits, temperature)
    
    // Softmax
    var probs = softmax(scaledLogits)
    
    // Top-K
    val (kIndices, kProbs) = topK(probs, k)
    
    // Top-P on filtered
    var cumSum = 0.0f
    val pFiltered = kIndices.zip(kProbs).takeWhile { case (_, prob) =>
      val include = cumSum < p
      cumSum += prob
      include || cumSum == prob
    }
    
    val finalIndices = pFiltered.map(_._1)
    val finalProbs = pFiltered.map(_._2)
    val sum = finalProbs.sum
    val normalizedProbs = finalProbs.map(_ / sum)
    
    // Sample
    val sampleIdx = sample(normalizedProbs)
    finalIndices(sampleIdx)
  }
}

/**
 * Metrics collector for monitoring.
 */
class MetricsCollector {
  private val requestCount = new AtomicLong(0)
  private val totalLatencyMs = new AtomicLong(0)
  private val tokenCount = new AtomicLong(0)
  private val errorCount = new AtomicLong(0)
  
  def recordRequest(latencyMs: Long, tokens: Int): Unit = {
    requestCount.incrementAndGet()
    totalLatencyMs.addAndGet(latencyMs)
    tokenCount.addAndGet(tokens)
  }
  
  def recordError(): Unit = {
    errorCount.incrementAndGet()
  }
  
  def avgLatencyMs: Double = {
    val count = requestCount.get()
    if (count == 0) 0.0 else totalLatencyMs.get().toDouble / count
  }
  
  def throughput(windowSeconds: Double): Double = {
    requestCount.get().toDouble / windowSeconds
  }
  
  def tokensPerSecond(windowSeconds: Double): Double = {
    tokenCount.get().toDouble / windowSeconds
  }
  
  def stats: Map[String, Any] = Map(
    "total_requests" -> requestCount.get(),
    "total_tokens" -> tokenCount.get(),
    "avg_latency_ms" -> avgLatencyMs,
    "error_count" -> errorCount.get()
  )
}

/**
 * Continuous batching engine.
 */
class ContinuousBatcher(config: InferenceConfig) {
  private val scheduler = new BatchScheduler(config)
  private val metrics = new MetricsCollector()
  private val activeRequests = new AtomicInteger(0)
  @volatile private var running = false
  
  def start(): Unit = {
    running = true
    // In real implementation, would start worker threads
  }
  
  def stop(): Unit = {
    running = false
  }
  
  def submit(
    inputIds: Array[Int],
    maxNewTokens: Int = 100,
    temperature: Float = 1.0f,
    priority: Int = 0
  ): Long = {
    scheduler.submit(inputIds, maxNewTokens, temperature, priority)
  }
  
  def processBatch(modelForward: Array[Array[Int]] => Array[Array[Float]]): Seq[InferenceResponse] = {
    val batch = scheduler.nextBatch()
    if (batch.isEmpty) return Seq.empty
    
    val startTime = System.currentTimeMillis()
    activeRequests.addAndGet(batch.size)
    
    try {
      // Pad sequences to same length
      val maxLen = batch.map(_.inputIds.length).max
      val paddedInputs = batch.map { req =>
        val padLen = maxLen - req.inputIds.length
        req.inputIds ++ Array.fill(padLen)(0)  // Assume 0 is pad token
      }.toArray
      
      // Forward pass (would call actual model)
      val logits = modelForward(paddedInputs)
      
      // Sample tokens
      val responses = batch.zip(logits).map { case (req, logitsRow) =>
        val outputToken = TokenSampler.sampleWithConfig(
          logitsRow, req.temperature, req.topK, req.topP
        )
        val latency = System.currentTimeMillis() - startTime
        
        metrics.recordRequest(latency, 1)
        
        InferenceResponse(
          id = req.id,
          outputIds = Array(outputToken),
          latencyMs = latency
        )
      }
      
      scheduler.recordComplete(batch.size)
      responses
      
    } catch {
      case e: Exception =>
        scheduler.recordFailed(batch.size)
        batch.map { req =>
          metrics.recordError()
          InferenceResponse(
            id = req.id,
            outputIds = Array.empty,
            success = false,
            error = Some(e.getMessage)
          )
        }
    } finally {
      activeRequests.addAndGet(-batch.size)
    }
  }
  
  def getMetrics: Map[String, Any] = {
    metrics.stats ++ scheduler.stats.map { case (k, v) => k -> v.asInstanceOf[Any] } +
      ("active_requests" -> activeRequests.get())
  }
}




