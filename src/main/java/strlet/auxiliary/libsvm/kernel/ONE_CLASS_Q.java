package strlet.auxiliary.libsvm.kernel;

import strlet.auxiliary.libsvm.SVMParameter;
import strlet.auxiliary.libsvm.SVMProblem;

public class ONE_CLASS_Q extends Kernel{

	private final Cache cache;
	private final double[] QD;

	public ONE_CLASS_Q(SVMProblem prob, SVMParameter param) {
		super(prob.l, prob.x, param);
		cache = new Cache(prob.l, (long) (param.cache_size * (1 << 20)));
		QD = new double[prob.l];
		for (int i = 0; i < prob.l; i++)
			QD[i] = kernel_function(i, i);
	}

	@Override
	public float[] get_Q(int i, int len) {
		float[][] data = new float[1][];
		int start, j;
		if ((start = cache.get_data(i, data, len)) < len) {
			for (j = start; j < len; j++)
				data[0][j] = (float) kernel_function(i, j);
		}
		return data[0];
	}

	@Override
	public double[] get_QD() {
		return QD;
	}

	@Override
	public void swap_index(int i, int j) {
		cache.swap_index(i, j);
		super.swap_index(i, j);
		do {
			double _x = QD[i];
			QD[i] = QD[j];
			QD[j] = _x;
		} while (false);
	}
}
