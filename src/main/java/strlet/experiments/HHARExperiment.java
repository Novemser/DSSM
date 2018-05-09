package strlet.experiments;

import com.novemser.HHAR;
import strlet.transferLearning.inductive.SingleSourceTransfer;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;

public class HHARExperiment extends AbstractExperiment {
  @Override
  public void runExperiment(SingleSourceTransfer[] models) throws Exception {
    DataSource ds = new DataSource("/home/novemser/Documents/Code/DSSM/src/main/resources/simple/load.csv");
    for (SingleSourceTransfer classifier : models) {
      Instances data = ds.getDataSet();
      data.setClassIndex(data.attribute("class").index());
//      data.deleteAttributeAt(data.attribute("id").index());
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
    HHARExperiment experiment = new HHARExperiment();
    experiment.runExperiment();
  }

}
