package org.apache.spark.ml.classification

/**
  * Wrapper class for convenient access of trees.
  * @param uid
  * @param _trees
  * @param numFeatures
  * @param numClasses
  */
class RichRandomForestClassificationModel(override val uid: String,
                                          val _trees: Array[DecisionTreeClassificationModel],
                                          override val numFeatures: Int,
                                          override val numClasses: Int)
  extends RandomForestClassificationModel(uid, _trees, numFeatures, numClasses) {
}
