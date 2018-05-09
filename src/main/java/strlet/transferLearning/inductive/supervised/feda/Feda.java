package strlet.transferLearning.inductive.supervised.feda;

import strlet.transferLearning.inductive.SingleSourceTransfer;
import strlet.transferLearning.inductive.supervised.SingleSourceInstanceTransfer;
import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class Feda extends SingleSourceInstanceTransfer {

	private Classifier baseClassifier = new ZeroR();

	private Instances transferedHeader = null;

	@Override
	public void buildModel(Instances source, Instances target) throws Exception {
		
		testWithFail(source, target);

		source = new Instances(source);
		source.deleteWithMissingClass();
		target = new Instances(target);
		target.deleteWithMissingClass();
		transferedHeader = expand(source);
		Instances training = new Instances(transferedHeader,
				source.numInstances() + target.numInstances());

		for (int index = 0; index < source.numInstances(); ++index) {
			Instance instance = source.instance(index);
			double[] before = instance.toDoubleArray();
			double[] after = new double[training.numAttributes()];
			for (int i = 0, j = 0; i < before.length; ++i, ++j) {
				if (instance.classIndex() == i) {
					after[after.length - 1] = before[i];
					--j;
				} else {
					after[3 * j] = before[i];
					after[3 * j + 1] = before[i];
					after[3 * j + 2] = Double.NaN;
				}
			}
			Instance newInstance = new Instance(1.0, after);
			newInstance.setDataset(training);
			training.add(newInstance);
		}

		for (int index = 0; index < target.numInstances(); ++index) {
			Instance instance = target.instance(index);
			double[] before = instance.toDoubleArray();
			double[] after = new double[training.numAttributes()];
			for (int i = 0, j = 0; i < before.length; ++i, ++j) {
				if (instance.classIndex() == i) {
					after[after.length - 1] = before[i];
					--j;
				} else {
					after[3 * j] = Double.NaN;
					after[3 * j + 1] = before[i];
					after[3 * j + 2] = before[i];
				}
			}
			Instance newInstance = new Instance(1.0, after);
			newInstance.setDataset(training);
			training.add(newInstance);
		}

		baseClassifier.buildClassifier(training);
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

		if (transferedHeader == null) {
			throw new Exception("No classifier trained yet!");
		}
		double[] before = instance.toDoubleArray();
		double[] after = new double[transferedHeader.numAttributes()];
		for (int i = 0, j = 0; i < before.length; ++i, ++j) {
			if (instance.classIndex() == i) {
				after[after.length - 1] = before[i];
				--j;
			} else {
				after[3 * j] = Double.NaN;
				after[3 * j + 1] = before[i];
				after[3 * j + 2] = before[i];
			}
		}
		instance = new Instance(1.0, after);
		instance.setDataset(transferedHeader);
		return baseClassifier.distributionForInstance(instance);
	}

	private Instances expand(Instances origin) {

		FastVector attInfo = new FastVector(origin.numAttributes() * 3 - 2);
		for (int i = 0; i < origin.numAttributes(); ++i) {
			if (i == origin.classIndex()) {
				continue;
			}
			Attribute attr = origin.attribute(i);
			if (attr.isNumeric()) {
				attInfo.addElement(new Attribute(attr.name() + "_s"));
				attInfo.addElement(new Attribute(attr.name() + "_st"));
				attInfo.addElement(new Attribute(attr.name() + "_t"));
			} else {
				FastVector values = new FastVector(attr.numValues());
				for (int j = 0; j < attr.numValues(); ++j) {
					values.addElement(attr.value(j));
				}
				attInfo.addElement(new Attribute(attr.name() + "_s", values));
				attInfo.addElement(new Attribute(attr.name() + "_st", values));
				attInfo.addElement(new Attribute(attr.name() + "_t", values));
			}
		}

		Attribute attr = origin.classAttribute();
		if (attr.isNumeric()) {
			attInfo.addElement(new Attribute(attr.name()));
		} else {
			FastVector values = new FastVector(attr.numValues());
			for (int j = 0; j < attr.numValues(); ++j) {
				values.addElement(attr.value(j));
			}
			attInfo.addElement(new Attribute(attr.name(), values));
		}

		Instances expanded = new Instances(origin.relationName(), attInfo, 0);
		expanded.setClassIndex(expanded.numAttributes() - 1);
		return expanded;

	}
	
	/**
	 * Set a base classifier for the FEDA algorithm
	 * @param baseClassifier The new base algorithm
	 * @throws Exception If any problem with the new base classifier occurred
	 */

	public void setBaseClassifier(Classifier baseClassifier) throws Exception {
		this.baseClassifier = Classifier.makeCopy(baseClassifier);
	}

	@Override
	public SingleSourceTransfer makeDuplicate() throws Exception {

		Feda other = new Feda();
		other.baseClassifier = Classifier.makeCopy(baseClassifier);
		return other;

	}

}
