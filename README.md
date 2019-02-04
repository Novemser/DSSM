# DSSM
An implementation of distributed structure expansion/reduction(SER) and structure transfer (STRUT) model transfer algorithm(see https://arxiv.org/pdf/1511.01258.pdf). Dataset used in the paper could be downloaded here. https://drive.google.com/file/d/19tFYaahJ2c_6El_ZVVIuXlVyXBgR-XrF/view?usp=sharing

## Requirements

- Spark 2.3.0
- Java 8
- Scala 2.11

## Usage
1. To reproduce expriment results in the original paper using DSSM, first prepare the dataset stated in the paper. A direct download link will be provided later.

2. DSSM is build as a Spark ML library, feel free to use it as a component inside your ML pipeline. For example, you could:
```scala
// Load data from any datasource that spark supports.
// Here we load data from HDFS cluster as an example.
val data = spark.read
    .option("header", "true")
    .option("inferSchema", true)
    .csv("hdfs://yourhdfsaddress:port/data/Phones_accelerometer_shuffle_del_10w.csv")
    .withColumnRenamed("gt", "class")
    .filter("class != 'null'")

// Then we define some commonly used ML pipeline components,
// more information could be found here http://spark.apache.org/docs/2.3.0/ml-guide.html
val trainLabelIndexer = new StringIndexer()
    .setHandleInvalid("skip")
    .setInputCol("class")
    .setOutputCol("label")
    .setStringOrderType("alphabetAsc")
    .fit(source)

val transferLabelIndexer = new StringIndexer()
    .setHandleInvalid("skip")
    .setInputCol("class")
    .setOutputCol("label")
    .setStringOrderType("alphabetAsc")
    .fit(target)

val trainAssembler = new VectorAssembler()
    .setInputCols(source.schema.map(_.name).filter(s => s != "class").toArray)
    .setOutputCol("features")

val transferAssembler = trainAssembler

// Next we define a SourceRandomForestClassifier, used as our source random forest
// which is going to be transferred later
val rf = new SourceRandomForestClassifier()
rf.setFeaturesCol { trainAssembler.getOutputCol }
    .setLabelCol { trainLabelIndexer.getOutputCol }
    .setMaxDepth(maxDepth)
    .setSeed(seed)
    .setNumTrees(numTrees)

// (Optional)You can choose which Impurity function to use and tune some hyper-parameters
// according to your need
treeType match {
    case TreeType.SER   => rf.setImpurity("entropy")
    case TreeType.STRUT => rf.setImpurity("entropy").setMinInfoGain(0.03) // prevent over fitting
    case TreeType.MIX   => rf.setImpurity("entropy")
}

// Combine components into a pipeline
val trainPipeline = new Pipeline()
    .setStages(Array(trainLabelIndexer, trainAssembler, rf))

// Train the source pipeline and calculate results
val srcAcc = Utils.trainAndTest(trainPipeline, source, test, berr, timer, "src")
if (srcOnly) {
    println(s"Src err:$srcAcc")
    return srcAcc
}

// Next we preform transfer learning steps on the trained model 

// Here you can shooes which classifier to use.
// SERClassifier stands for distributed SER
// STRUTClassifier stands for distributed STRUT
// MixClassifier stands for distributed MIX
val classifier = treeType match {
    case TreeType.SER   => new SERClassifier(rf.model)
    case TreeType.STRUT => new STRUTClassifier(rf.model)
    case TreeType.MIX   => new MixClassifier(rf.model)
}

classifier
    .setFeaturesCol { trainAssembler.getOutputCol }
    .setLabelCol { trainLabelIndexer.getOutputCol }
    .setMaxDepth { maxDepth }
    .setSeed { seed }

// (Optional)Again tune some hyper-parameters
treeType match {
    case TreeType.SER   => classifier.setImpurity("entropy")
    case TreeType.STRUT => classifier.setImpurity("entropy")
    case TreeType.MIX   => classifier.setImpurity("entropy")
}

val transferPipeline = new Pipeline()
    .setStages(Array(transferLabelIndexer, transferAssembler, classifier))

// Finally, perform the actual transfer learning procedure and print the result
val transferAcc = Utils.trainAndTest(transferPipeline, target, test, berr, timer, "transfer")
println(s"SrcOnly err:$srcAcc, $treeType err:$transferAcc")

```

3. More example usage could be found in HHAR.scala and DSSM.scala
