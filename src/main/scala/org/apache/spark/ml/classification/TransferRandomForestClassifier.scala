package org.apache.spark.ml.classification
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.tree.impl.{RandomForest, RichDecisionTreeClassificationModel, TransferRandomForest}
import org.apache.spark.ml.util.{Instrumentation, MetadataUtils}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}

class TransferRandomForestClassifier
  extends RandomForestClassifier {

  var splitFunction: LabeledPoint => Boolean = null

  protected override def train(dataset: Dataset[_]): RandomForestClassificationModel = {
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val numClasses: Int = getNumClasses(dataset)

    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".train() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset, numClasses)
    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses, OldAlgo.Classification, getOldImpurity)

    val src = oldDataset.filter(splitFunction)
    val tgt = oldDataset.filter(!splitFunction(_))
    println(s"Feature size: ${oldDataset.first().features.size}")
    println(s"Src count:${src.count()}, tgt count:${tgt.count()}")

    val instr = Instrumentation.create(this, oldDataset)
    instr.logParams(labelCol, featuresCol, predictionCol, probabilityCol, rawPredictionCol,
      impurity, numTrees, featureSubsetStrategy, maxDepth, maxBins, maxMemoryInMB, minInfoGain,
      minInstancesPerNode, seed, subsamplingRate, thresholds, cacheNodeIds, checkpointInterval)

    val trees = TransferRandomForest
      .run(src, strategy, getNumTrees, getFeatureSubsetStrategy, getSeed, Some(instr))
      .map(_.asInstanceOf[RichDecisionTreeClassificationModel])

    val transferTrees = TransferRandomForest
      .transferModels(trees, tgt, strategy, getNumTrees, getFeatureSubsetStrategy, getSeed, Some(instr))

    val numFeatures = oldDataset.first().features.size
    val m = new RandomForestClassificationModel(uid, transferTrees.map(_.asInstanceOf[DecisionTreeClassificationModel]),
      numFeatures, numClasses)
    instr.logSuccess(m)
    m
  }
}
