package org.apache.spark.ml.classification
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.tree.impl.TransferRandomForest
import org.apache.spark.ml.util.{Instrumentation, MetadataUtils}
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

class SourceRandomForestClassifier extends RandomForestClassifier {

  var model: RichRandomForestClassificationModel = _

  protected override def train(dataset: Dataset[_]): RandomForestClassificationModel = {
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val numClasses: Int = getNumClasses(dataset)

    if (isDefined(thresholds)) {
      require(
        $(thresholds).length == numClasses,
        this.getClass.getSimpleName +
          ".train() called with non-matching numClasses and thresholds.length." +
          s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}"
      )
    }

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset, numClasses)
    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses, OldAlgo.Classification, getOldImpurity)

    val instr = Instrumentation.create(this, oldDataset)
    instr.logParams(
      labelCol,
      featuresCol,
      predictionCol,
      probabilityCol,
      rawPredictionCol,
      impurity,
      numTrees,
      featureSubsetStrategy,
      maxDepth,
      maxBins,
      maxMemoryInMB,
      minInfoGain,
      minInstancesPerNode,
      seed,
      subsamplingRate,
      thresholds,
      cacheNodeIds,
      checkpointInterval
    )

    val trees = TransferRandomForest
      .run(oldDataset, strategy, getNumTrees, getFeatureSubsetStrategy, getSeed, Some(instr))
      .map(_.asInstanceOf[DecisionTreeClassificationModel])

    val numFeatures = oldDataset.first().features.size
    model = new RichRandomForestClassificationModel(uid, trees, numFeatures, numClasses)
    instr.logSuccess(model)
    model
  }
}
