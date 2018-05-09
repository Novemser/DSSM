package strlet.experiments;

import strlet.transferLearning.inductive.SingleSourceTransfer;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class CustomExperiment extends AbstractExperiment {

  @Override
  public void runExperiment(SingleSourceTransfer[] models) throws Exception {
    DataSource ds = new DataSource("/home/novemser/Documents/Code/Java/strlet/testData/load.arff");
    for (SingleSourceTransfer classifier : models) {
      Instances data = ds.getDataSet();
//      data.deleteAttributeAt(0);
      Instances dup = new Instances(data);
      dup.setClassIndex(dup.attribute("class").index());
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
    CustomExperiment customExperiment = new CustomExperiment();
    customExperiment.runExperiment();
  }
}
