package strlet.auxiliary;

import java.util.Random;

import weka.classifiers.RandomizableSingleClassifierEnhancer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.unsupervised.instance.Randomize;

public class BalancedClassifier extends RandomizableSingleClassifierEnhancer {

	private static final long serialVersionUID = 5042207483651438875L;

	@Override
	public void buildClassifier(Instances data) throws Exception {

		data = new Instances(data);
		data.deleteWithMissingClass();
		int counter[] = new int[data.numClasses()];
		for (int index = 0; index < data.numInstances(); ++index) {
			int v = (int) Math.round(data.instance(index).classValue());
			++counter[v];
		}
		int min = counter[Utils.minIndex(counter)];
		Instances balanced = new Instances(data,min*counter.length);
		
		Random rand = new Random(getSeed());
		for (int classI=0; classI<counter.length; ++classI) {
			int[] indeces = new int[counter[classI]];
			int c = 0;
			for (int index = 0; index < data.numInstances(); ++index) {
				int v = (int) Math.round(data.instance(index).classValue());
				if(v == classI){
					indeces[c++] = index;
				}
			}
			for (int i=0; i<min; ++i) {
				int n = rand.nextInt(counter[classI]);
				int index = indeces[n];
				indeces[n] = indeces[counter[classI]-1];
				--counter[classI];
				Instance instance = data.instance(index);
				balanced.add(instance);
			}
		}
		balanced.randomize(rand);

		if (Randomize.class.isAssignableFrom(m_Classifier.getClass())) {
			Randomize.class.cast(m_Classifier).setRandomSeed(getSeed());
		}
		m_Classifier.buildClassifier(balanced);

	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		return m_Classifier.distributionForInstance(instance);
	}

}
