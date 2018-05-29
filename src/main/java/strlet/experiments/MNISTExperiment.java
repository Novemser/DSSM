package strlet.experiments;

import strlet.transferLearning.inductive.SingleSourceTransfer;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class MNISTExperiment extends AbstractExperiment {
  @Override
  public void runExperiment(SingleSourceTransfer[] models) throws Exception {
    ConverterUtils.DataSource ds = new ConverterUtils.DataSource("/home/novemser/mnist.libsvm");
    Instances data = ds.getDataSet();
//    data.setClassIndex(784);
    for (SingleSourceTransfer classifier : models) {
      Instances dup = new Instances(data);
      double err = runExperiment(classifier, dup, new Instances(dup));
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
    MNISTExperiment ex = new MNISTExperiment();
    ex.runExperiment();
  }
}
