package strlet.auxiliary.libsvm.kernel;

import strlet.auxiliary.libsvm.KernelType;
import strlet.auxiliary.libsvm.SVMNode;
import strlet.auxiliary.libsvm.SVMParameter;

/**
Kernel evaluation

the static method k_function is for doing single kernel evaluation
the constructor of Kernel prepares to calculate the l*l kernel matrix
the member function get_Q is for getting one column from the Q Matrix
*/
public abstract class Kernel {
	private SVMNode[][] x;
	private final double[] x_square;

	// svm_parameter
	private final KernelType kernel_type;
	private final int degree;
	private final double gamma;
	private final double coef0;

	public abstract float[] get_Q(int column, int len);

	public abstract double[] get_QD();

	public void swap_index(int i, int j) {
		do {
			SVMNode[] _x = x[i];
			x[i] = x[j];
			x[j] = _x;
		} while (false);
		if (x_square != null)
			do {
				double _x = x_square[i];
				x_square[i] = x_square[j];
				x_square[j] = _x;
			} while (false);
	}

	private static double powi(double base, int times) {
		double tmp = base, ret = 1.0;

		for (int t = times; t > 0; t /= 2) {
			if (t % 2 == 1)
				ret *= tmp;
			tmp = tmp * tmp;
		}
		return ret;
	}

	protected double kernel_function(int i, int j) {
		switch (kernel_type) {
		case LINEAR:
			return dot(x[i], x[j]);
		case POLYNOMIAL:
			return powi(gamma * dot(x[i], x[j]) + coef0, degree);
		case RBF:
			return Math.exp(-gamma
					* (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
		case SIGMOID:
			return Math.tanh(gamma * dot(x[i], x[j]) + coef0);
		case PRECOMPUTED:
			return x[i][(int) (x[j][0].value)].value;
		default:
			return 0; // java
		}
	}

	protected Kernel(int l, SVMNode[][] x_, SVMParameter param) {
		this.kernel_type = param.kernel_type;
		this.degree = param.degree;
		this.gamma = param.gamma;
		this.coef0 = param.coef0;

		x = (SVMNode[][]) x_.clone();

		if (kernel_type.equals(KernelType.RBF)) {
			x_square = new double[l];
			for (int i = 0; i < l; i++)
				x_square[i] = dot(x[i], x[i]);
		} else
			x_square = null;
	}

	private static double dot(SVMNode[] x, SVMNode[] y) {
		double sum = 0;
		int xlen = x.length;
		int ylen = y.length;
		int i = 0;
		int j = 0;
		while (i < xlen && j < ylen) {
			if (x[i].index == y[j].index)
				sum += x[i++].value * y[j++].value;
			else {
				if (x[i].index > y[j].index)
					++j;
				else
					++i;
			}
		}
		return sum;
	}

	public static double k_function(SVMNode[] x, SVMNode[] y, SVMParameter param) {
		switch (param.kernel_type) {
		case LINEAR:
			return dot(x, y);
		case POLYNOMIAL:
			return powi(param.gamma * dot(x, y) + param.coef0, param.degree);
		case RBF: {
			double sum = 0;
			int xlen = x.length;
			int ylen = y.length;
			int i = 0;
			int j = 0;
			while (i < xlen && j < ylen) {
				if (x[i].index == y[j].index) {
					double d = x[i++].value - y[j++].value;
					sum += d * d;
				} else if (x[i].index > y[j].index) {
					sum += y[j].value * y[j].value;
					++j;
				} else {
					sum += x[i].value * x[i].value;
					++i;
				}
			}

			while (i < xlen) {
				sum += x[i].value * x[i].value;
				++i;
			}

			while (j < ylen) {
				sum += y[j].value * y[j].value;
				++j;
			}

			return Math.exp(-param.gamma * sum);
		}
		case SIGMOID:
			return Math.tanh(param.gamma * dot(x, y) + param.coef0);
		case PRECOMPUTED:
			return x[(int) (y[0].value)].value;
		default:
			return 0; // java
		}
	}
}
