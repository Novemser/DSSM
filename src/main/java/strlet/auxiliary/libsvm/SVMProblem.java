package strlet.auxiliary.libsvm;

import java.io.Serializable;

public class SVMProblem implements Serializable {

	private static final long serialVersionUID = -4451389443706847272L;

	public int l;
	public double[] y;
	public SVMNode[][] x;
}
