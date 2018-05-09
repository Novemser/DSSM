package strlet.experiments;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.Random;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import strlet.transferLearning.inductive.SingleSourceTransfer;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class OfficeCaltechExperiment extends AbstractExperiment {

	private static final String[] domains = { "amazon", "Caltech10", "dslr",
			"webcam" };

	public void runExperiment(SingleSourceTransfer[] classifiers)
			throws Exception {

		ZipFile zip = new ZipFile(new File("testData" + File.separator
				+ "office-caltech.zip"));
		Instances[] dataDomains = new Instances[4];
		for (int i = 0; i < 4; ++i) {
			dataDomains[i] = loadData(zip, domains[i]);
		}
		zip.close();

		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 4; ++j) {
				if (i == j) {
					continue;
				}
				System.out.println(domains[i] + "->" + domains[j]);
				for (SingleSourceTransfer classifier : classifiers) {
					Instances source = new Instances(dataDomains[i]);
					Instances target = new Instances(dataDomains[j]);
					double tot = 0;
					for (int k = 0; k < 2; ++k) {
						source.randomize(new Random(SEED * (i + 1)));
						target.randomize(new Random(SEED * (i + 1)));
						double err = runExperiment(classifier, source, target);
						tot += err;
					}
					System.out.println(ToPerc(tot / 2));
				}
			}
		}

	}

	private Instances loadData(ZipFile zip, String domainName) throws Exception {

		ZipEntry dataEntry = zip.getEntry(domainName + "_SURF_L10_fts.csv");
		ZipEntry labelsEntry = zip
				.getEntry(domainName + "_SURF_L10_labels.csv");

		BufferedReader br = new BufferedReader(new InputStreamReader(
				zip.getInputStream(labelsEntry)));
		int total = 0;
		while (br.readLine() != null) {
			++total;
		}
		br.close();

		BufferedReader dataReader = new BufferedReader(new InputStreamReader(
				zip.getInputStream(dataEntry)));
		BufferedReader labellsReader = new BufferedReader(
				new InputStreamReader(zip.getInputStream(labelsEntry)));
		FastVector attInfo = new FastVector(801);
		for (int i = 0; i < 800; ++i) {
			Attribute attr = new Attribute("attr" + i);
			attInfo.addElement(attr);
		}
		FastVector classValues = new FastVector(10);
		for (int i = 1; i <= 10; ++i) {
			classValues.addElement("" + i);
		}
		Attribute classAttr = new Attribute("class", classValues);
		attInfo.addElement(classAttr);
		Instances instances = new Instances(domainName, attInfo, total);
		instances.setClassIndex(800);

		for (int i = 0; i < total; ++i) {
			String[] split = dataReader.readLine().split(",");
			double[] values = new double[801];
			for (int j = 0; j < split.length; ++j) {
				values[j] = Double.valueOf(split[j]);
			}
			String label = labellsReader.readLine();
			values[800] = classAttr.indexOfValue(label);
			Instance instance = new Instance(1.0, values);
			instances.add(instance);
		}
		dataReader.close();
		labellsReader.close();
		return instances;
	}

	private double runExperiment(SingleSourceTransfer classifier,
			Instances source, Instances target) throws Exception {

		target = new Instances(target);
		target.stratify(20);
		double total = 0;
		for (int i = 0; i < 20; ++i) {
			Instances train = target.testCV(20, i);
			Instances test = target.trainCV(20, i);
			SingleSourceTransfer dup = classifier.makeDuplicate();
			dup.buildModel(source, train);
			total += err(dup, test);
		}
		return total / 20;

	}

	public static void main(String[] args) throws Exception {
		OfficeCaltechExperiment experiment = new OfficeCaltechExperiment();
		experiment.runExperiment();
	}

}
