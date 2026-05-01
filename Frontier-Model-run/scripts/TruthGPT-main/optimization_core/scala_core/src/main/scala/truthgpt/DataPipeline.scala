/**
 * TruthGPT Scala Core - Data Pipeline
 * 
 * High-performance data processing pipeline for LLM training.
 * Uses Apache Spark for distributed processing.
 * 
 * Features:
 * - Distributed JSONL processing
 * - Tokenization with parallel execution
 * - Data shuffling and batching
 * - Efficient memory management
 */
package truthgpt

import scala.collection.mutable.ArrayBuffer
import scala.util.{Try, Success, Failure}
import java.io.{BufferedReader, FileReader, BufferedWriter, FileWriter}
import java.util.concurrent.{Executors, LinkedBlockingQueue, TimeUnit}
import java.util.concurrent.atomic.{AtomicLong, AtomicBoolean}

/**
 * Configuration for data pipeline.
 */
case class DataPipelineConfig(
  batchSize: Int = 32,
  maxSeqLen: Int = 2048,
  numWorkers: Int = 4,
  shuffleBuffer: Int = 10000,
  prefetchSize: Int = 100
)

/**
 * Data sample for training.
 */
case class DataSample(
  inputIds: Array[Int],
  attentionMask: Array[Int],
  labels: Option[Array[Int]] = None
)

/**
 * Training batch.
 */
case class TrainingBatch(
  inputIds: Array[Array[Int]],
  attentionMask: Array[Array[Int]],
  labels: Array[Array[Int]],
  batchSize: Int
)

/**
 * Simple JSON parser for JSONL files.
 */
object JsonParser {
  /**
   * Parse a simple JSON object with string values.
   */
  def parseJsonLine(line: String): Map[String, String] = {
    val trimmed = line.trim
    if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
      return Map.empty
    }
    
    val content = trimmed.drop(1).dropRight(1)
    val pairs = content.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")
    
    pairs.flatMap { pair =>
      val parts = pair.split(":", 2)
      if (parts.length == 2) {
        val key = parts(0).trim.stripPrefix("\"").stripSuffix("\"")
        val value = parts(1).trim.stripPrefix("\"").stripSuffix("\"")
        Some(key -> value)
      } else None
    }.toMap
  }
}

/**
 * Tokenizer interface.
 */
trait Tokenizer {
  def encode(text: String): Array[Int]
  def decode(ids: Array[Int]): String
  def vocabSize: Int
  def padTokenId: Int
  def eosTokenId: Int
}

/**
 * Simple byte-level tokenizer for demonstration.
 */
class ByteTokenizer extends Tokenizer {
  override def encode(text: String): Array[Int] = {
    text.getBytes("UTF-8").map(_.toInt & 0xFF)
  }
  
  override def decode(ids: Array[Int]): String = {
    new String(ids.map(_.toByte), "UTF-8")
  }
  
  override def vocabSize: Int = 256
  override def padTokenId: Int = 0
  override def eosTokenId: Int = 1
}

/**
 * Data loader with prefetching.
 */
class DataLoader(
  filePath: String,
  tokenizer: Tokenizer,
  config: DataPipelineConfig
) {
  private val prefetchQueue = new LinkedBlockingQueue[TrainingBatch](config.prefetchSize)
  private val running = new AtomicBoolean(false)
  private val samplesProcessed = new AtomicLong(0)
  private val batchesCreated = new AtomicLong(0)
  
  private var prefetchThread: Thread = _
  
  /**
   * Start prefetching batches in background.
   */
  def start(): Unit = {
    if (running.compareAndSet(false, true)) {
      prefetchThread = new Thread(() => prefetchLoop())
      prefetchThread.setDaemon(true)
      prefetchThread.start()
    }
  }
  
  /**
   * Stop the data loader.
   */
  def stop(): Unit = {
    running.set(false)
    if (prefetchThread != null) {
      prefetchThread.interrupt()
    }
  }
  
  /**
   * Get the next batch, blocking if necessary.
   */
  def nextBatch(): Option[TrainingBatch] = {
    if (!running.get() && prefetchQueue.isEmpty) None
    else {
      Option(prefetchQueue.poll(1, TimeUnit.SECONDS))
    }
  }
  
  /**
   * Get the next batch with timeout.
   */
  def nextBatch(timeoutMs: Long): Option[TrainingBatch] = {
    Option(prefetchQueue.poll(timeoutMs, TimeUnit.MILLISECONDS))
  }
  
  private def prefetchLoop(): Unit = {
    val reader = new BufferedReader(new FileReader(filePath))
    val buffer = ArrayBuffer[DataSample]()
    
    try {
      var line = reader.readLine()
      while (line != null && running.get()) {
        // Parse and tokenize
        val json = JsonParser.parseJsonLine(line)
        json.get("text").foreach { text =>
          val sample = tokenizeText(text)
          buffer += sample
          samplesProcessed.incrementAndGet()
          
          // Create batch when buffer is full
          if (buffer.size >= config.batchSize) {
            val batch = createBatch(buffer.take(config.batchSize).toArray)
            prefetchQueue.put(batch)
            batchesCreated.incrementAndGet()
            buffer.remove(0, config.batchSize)
          }
        }
        
        line = reader.readLine()
      }
      
      // Handle remaining samples
      if (buffer.nonEmpty) {
        val batch = createBatch(buffer.toArray)
        prefetchQueue.put(batch)
        batchesCreated.incrementAndGet()
      }
      
    } finally {
      reader.close()
    }
  }
  
  private def tokenizeText(text: String): DataSample = {
    val ids = tokenizer.encode(text)
    val truncated = ids.take(config.maxSeqLen)
    val mask = Array.fill(truncated.length)(1)
    
    DataSample(
      inputIds = truncated,
      attentionMask = mask,
      labels = Some(truncated)  // For language modeling
    )
  }
  
  private def createBatch(samples: Array[DataSample]): TrainingBatch = {
    val batchSize = samples.length
    val maxLen = samples.map(_.inputIds.length).max
    
    // Pad sequences
    val inputIds = samples.map { s =>
      val padLen = maxLen - s.inputIds.length
      s.inputIds ++ Array.fill(padLen)(tokenizer.padTokenId)
    }
    
    val attentionMask = samples.map { s =>
      val padLen = maxLen - s.attentionMask.length
      s.attentionMask ++ Array.fill(padLen)(0)
    }
    
    val labels = samples.map { s =>
      val ids = s.labels.getOrElse(s.inputIds)
      val padLen = maxLen - ids.length
      ids ++ Array.fill(padLen)(-100)  // -100 for ignore index
    }
    
    TrainingBatch(inputIds, attentionMask, labels, batchSize)
  }
  
  def stats: Map[String, Long] = Map(
    "samples_processed" -> samplesProcessed.get(),
    "batches_created" -> batchesCreated.get(),
    "queue_size" -> prefetchQueue.size()
  )
}

/**
 * Parallel data processor using thread pool.
 */
class ParallelProcessor(numWorkers: Int) {
  private val executor = Executors.newFixedThreadPool(numWorkers)
  
  /**
   * Process items in parallel.
   */
  def map[A, B](items: Seq[A])(f: A => B): Seq[B] = {
    import scala.concurrent.{Await, Future}
    import scala.concurrent.duration._
    
    implicit val ec = scala.concurrent.ExecutionContext.fromExecutor(executor)
    
    val futures = items.map(item => Future(f(item)))
    Await.result(Future.sequence(futures), Duration.Inf)
  }
  
  /**
   * Process with batch-level parallelism.
   */
  def mapBatches[A, B](items: Seq[A], batchSize: Int)(f: Seq[A] => Seq[B]): Seq[B] = {
    val batches = items.grouped(batchSize).toSeq
    map(batches)(f).flatten
  }
  
  def shutdown(): Unit = {
    executor.shutdown()
    executor.awaitTermination(30, TimeUnit.SECONDS)
  }
}

/**
 * Shuffle buffer for streaming data.
 */
class ShuffleBuffer[A](capacity: Int) {
  private val buffer = ArrayBuffer[A]()
  private val rng = new scala.util.Random()
  
  def add(item: A): Option[A] = {
    buffer += item
    if (buffer.size > capacity) {
      Some(pop())
    } else None
  }
  
  def pop(): A = {
    val idx = rng.nextInt(buffer.size)
    val item = buffer(idx)
    buffer.remove(idx)
    item
  }
  
  def drain(): Seq[A] = {
    val result = rng.shuffle(buffer.toSeq)
    buffer.clear()
    result
  }
  
  def size: Int = buffer.size
}

/**
 * Data pipeline orchestrator.
 */
class DataPipeline(config: DataPipelineConfig) {
  private val processor = new ParallelProcessor(config.numWorkers)
  
  /**
   * Process a JSONL file and return batches.
   */
  def procesFile(
    filePath: String,
    tokenizer: Tokenizer
  ): Iterator[TrainingBatch] = {
    val loader = new DataLoader(filePath, tokenizer, config)
    loader.start()
    
    new Iterator[TrainingBatch] {
      private var nextBatchOpt: Option[TrainingBatch] = None
      private var exhausted = false
      
      override def hasNext: Boolean = {
        if (exhausted) return false
        if (nextBatchOpt.isDefined) return true
        
        nextBatchOpt = loader.nextBatch(5000)
        if (nextBatchOpt.isEmpty) {
          exhausted = true
          loader.stop()
          false
        } else true
      }
      
      override def next(): TrainingBatch = {
        if (!hasNext) throw new NoSuchElementException
        val batch = nextBatchOpt.get
        nextBatchOpt = None
        batch
      }
    }
  }
  
  /**
   * Tokenize texts in parallel.
   */
  def tokenizeParallel(texts: Seq[String], tokenizer: Tokenizer): Seq[Array[Int]] = {
    processor.map(texts)(tokenizer.encode)
  }
  
  def shutdown(): Unit = {
    processor.shutdown()
  }
}
