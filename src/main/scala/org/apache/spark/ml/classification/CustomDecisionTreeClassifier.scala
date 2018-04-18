package org.apache.spark.ml.classification

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tree.RandomForestClassifierParams
import org.apache.spark.ml.tree.impl.{RichDecisionTreeClassificationModel, SERTransfer, TransferRandomForest}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, Instrumentation, MetadataUtils}
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo, Strategy => OldStrategy}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

class CustomDecisionTreeClassifier(val uid: String)
    extends ProbabilisticClassifier[
      Vector,
      CustomDecisionTreeClassifier,
      DecisionTreeClassificationModel
    ]
    with RandomForestClassifierParams
    with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("decTree"))

  var splitFunction: LabeledPoint => Boolean = null

  override def copy(extra: ParamMap): CustomDecisionTreeClassifier = {
    this
  }

  override protected def train(dataset: Dataset[_]): DecisionTreeClassificationModel = {
    println("Test train")
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val numClasses: Int = getNumClasses(dataset)
    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset, numClasses)
    val strategy = getOldStrategy(categoricalFeatures, numClasses)
    val instr = Instrumentation.create(this, oldDataset)
    instr.logParams(params: _*)

//    val Array(src, tgt) = oldDataset.randomSplit(Array(0.2, 0.8))
    val src = oldDataset.filter(splitFunction)
    val tgt = oldDataset.filter(!splitFunction(_))
    println(s"Feature size: ${oldDataset.first().features.size}")
    println(s"Src count:${src.count()}, tgt count:${tgt.count()}")

//    src.foreach((point: LabeledPoint) => {
//      val vals = point.features.toArray
//      println(vals(9))
//    })

    val trees = TransferRandomForest
      .run(src, strategy, getNumTrees, getFeatureSubsetStrategy, getSeed, Some(instr))
      .map(_.asInstanceOf[DecisionTreeClassificationModel])
//    val numFeatures = oldDataset.first().features.size
//    val m = new RandomForestClassificationModel(uid, trees, numFeatures, numClasses)
//    instr.logSuccess(m)
//    m

    //    println(categoricalFeatures)
//    println(numClasses)
    val m = trees.head
    instr.logSuccess(m)
    println("Transfer----------------------------------------------")

    val res = SERTransfer.transfer(
      m.asInstanceOf[RichDecisionTreeClassificationModel],
      tgt,
      strategy,
      numTrees = 1,
      featureSubsetStrategy = "all",
      seed = $(seed),
      instr = Some(instr),
      parentUID = Some(uid)
    )
    res
  }

  private[ml] def getOldStrategy(categoricalFeatures: Map[Int, Int], numClasses: Int): OldStrategy = {
    super.getOldStrategy(
      categoricalFeatures,
      numClasses,
      OldAlgo.Classification,
      getOldImpurity,
      subsamplingRate = 1.0
    )
  }
}

//class DTCModel private[ml](
//                          override val uid: String
//
//                          )
//extends ProbabilisticClassificationModel[Vector, DTCModel]
//with DecisionTreeModel with DecisionTreeClassifierParams with MLWritable with Serializable {
//
//}
