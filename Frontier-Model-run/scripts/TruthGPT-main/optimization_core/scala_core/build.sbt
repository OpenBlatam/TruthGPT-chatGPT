// TruthGPT Scala Core Build Configuration

name := "truthgpt-scala-core"
version := "0.1.0"
scalaVersion := "2.13.12"

// Java target version
javacOptions ++= Seq("-source", "11", "-target", "11")
scalacOptions ++= Seq(
  "-deprecation",
  "-encoding", "UTF-8",
  "-feature",
  "-unchecked",
  "-Xfatal-warnings",
  "-Xlint:_",
  "-Ywarn-dead-code",
  "-Ywarn-numeric-widen",
  "-Ywarn-value-discard"
)

// Dependencies
libraryDependencies ++= Seq(
  // Apache Spark
  "org.apache.spark" %% "spark-core" % "3.5.0" % "provided",
  "org.apache.spark" %% "spark-sql" % "3.5.0" % "provided",
  "org.apache.spark" %% "spark-mllib" % "3.5.0" % "provided",
  
  // Akka for streaming
  "com.typesafe.akka" %% "akka-actor-typed" % "2.8.5",
  "com.typesafe.akka" %% "akka-stream" % "2.8.5",
  "com.typesafe.akka" %% "akka-http" % "10.5.3",
  
  // Cats Effect for functional effects
  "org.typelevel" %% "cats-effect" % "3.5.2",
  "org.typelevel" %% "cats-core" % "2.10.0",
  
  // ZIO
  "dev.zio" %% "zio" % "2.0.19",
  "dev.zio" %% "zio-streams" % "2.0.19",
  
  // JSON processing
  "io.circe" %% "circe-core" % "0.14.6",
  "io.circe" %% "circe-generic" % "0.14.6",
  "io.circe" %% "circe-parser" % "0.14.6",
  
  // gRPC
  "io.grpc" % "grpc-netty" % "1.59.0",
  "io.grpc" % "grpc-protobuf" % "1.59.0",
  "io.grpc" % "grpc-stub" % "1.59.0",
  
  // Metrics
  "io.dropwizard.metrics" % "metrics-core" % "4.2.21",
  "io.prometheus" % "simpleclient" % "0.16.0",
  
  // Logging
  "ch.qos.logback" % "logback-classic" % "1.4.11",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.9.5",
  
  // Testing
  "org.scalatest" %% "scalatest" % "3.2.17" % Test,
  "org.scalatestplus" %% "scalacheck-1-17" % "3.2.17.0" % Test
)

// Assembly settings for fat JAR
assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case "reference.conf" => MergeStrategy.concat
  case x => MergeStrategy.first
}

// Test settings
Test / parallelExecution := false
Test / fork := true

// Enable JMH benchmarks
enablePlugins(JmhPlugin)
