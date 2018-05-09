package strlet.experiments;

import com.novemser.HHAR;
import strlet.auxiliary.ThreadPool;
import strlet.transferLearning.inductive.SingleSourceTransfer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;

public class HHARExperiment extends AbstractExperiment {
  @Override
  public void runExperiment(SingleSourceTransfer[] models) throws Exception {
    DataSource ds = new DataSource(
        "/home/novemser/data/ActivityRecognitionExp/Phones_accelerometer_shuffle_del_10w.csv");


    Instances data = ds.getDataSet();
    data.setClassIndex(data.attribute("gt").index());
//    data.deleteAttributeAt(data.attribute("Model").index());
    int attIndex = data.attribute("Model").index();
    Instances s3 = new Instances(data, data.numInstances());
    Instances samsungold = new Instances(data, data.numInstances());
    Instances s3mini = new Instances(data, data.numInstances());
    Instances nexus4 = new Instances(data, data.numInstances());
    for (int index = 0; index < data.numInstances(); ++index) {
      Instance instance = data.instance(index);
//      System.out.println("instance.classValue():" + instance.classValue());
      //s3,samsungold,s3mini,nexus4
      double val = instance.value(attIndex);
      if (Utils.eq(0, val)) {
        s3.add(instance);
      } else if (Utils.eq(1, val)) {
        samsungold.add(instance);
      } else if (Utils.eq(2, val)) {
        s3mini.add(instance);
      } else if (Utils.eq(3, val)) {
        nexus4.add(instance);
      }
    }
    nexus4.deleteAttributeAt(attIndex);
    s3.deleteAttributeAt(attIndex);
    s3mini.deleteAttributeAt(attIndex);
    samsungold.deleteAttributeAt(attIndex);
    for (SingleSourceTransfer classifier : models) {
      Instances dup = new Instances(data);
      double err = runExperiment(classifier, nexus4, new Instances(s3));
      System.out.println("Error:" + err);
    }
  }

  private double runExperiment(SingleSourceTransfer classifier,
                               Instances source, Instances target) throws Exception {

    target = new Instances(target);

    double total = 0;
    SingleSourceTransfer dup = classifier.makeDuplicate();

    dup.buildModel(source, target);

    total += err(dup, target);

    return total;
  }

  public static void main(String[] args) throws Exception {
    HHARExperiment experiment = new HHARExperiment();
    ThreadPool.initialize(16);
    try {
      experiment.runExperiment();
    } finally {
      ThreadPool.shutdown();
    }
//    args = new String[]{
//        "/home/novemser/data/ActivityRecognitionExp/Phones_accelerometer_shuffle_del_1w.csv",
//        "/home/novemser/data/ActivityRecognitionExp/Phones_accelerometer_shuffle_del_1w.arff"
//    };
//    CSVLoader loader = new CSVLoader();
//    loader.setSource(new File(args[0]));
//    Instances data = loader.getDataSet();
//
//    // save ARFF
//    ArffSaver saver = new ArffSaver();
//    saver.setInstances(data);
//    saver.setFile(new File(args[1]));
//    saver.setDestination(new File(args[1]));
//    saver.writeBatch();
  }

}
