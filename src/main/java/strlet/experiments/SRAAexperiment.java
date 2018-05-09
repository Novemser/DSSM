package strlet.experiments;

import java.io.File;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Random;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

public class SRAAexperiment extends TextExperiment {

	private static final String[][] classes = { { "auto", "aviation" },
			{ "real", "sim" } };

	public SRAAexperiment(String a, String b) {
		super(a, b);
	}

	@Override
	protected boolean isClass(String name, String classA) {
		return name.contains(classA);
	}

	@Override
	protected LinkedList<String> getFileNames(int fold) throws Exception {
		if ((fold < 0) || (fold >= 2)) {
			throw new Exception(
					"More domains attempted than currently supported");
		}

		ZipFile zip = new ZipFile(ZIP_FILE);
		Enumeration<? extends ZipEntry> entries = zip.entries();
		HashMap<String, Integer> sourceSubjects = new HashMap<String, Integer>();
		HashMap<String, Integer> targetSubjects = new HashMap<String, Integer>();
		while (entries.hasMoreElements()) {
			ZipEntry entry = entries.nextElement();
			if (entry.isDirectory()) {
				String name = entry.getName().substring(0,
						entry.getName().length() - 1);
				if (a.equals("auto")) {
					if (name.contains("sim")) {
						sourceSubjects.put(name, sourceSubjects.size());
					} else {
						targetSubjects.put(name, targetSubjects.size());
					}
				} else {
					if (name.contains("aviation")) {
						sourceSubjects.put(name, sourceSubjects.size());
					} else {
						targetSubjects.put(name, targetSubjects.size());
					}
				}
			}
		}

		HashMap<String, LinkedList<String>> allNames = new HashMap<String, LinkedList<String>>();
		entries = zip.entries();
		while (entries.hasMoreElements()) {
			ZipEntry entry = entries.nextElement();
			String name = entry.getName().substring(0,
					entry.getName().indexOf("/"));
			if ((fold == 0) && targetSubjects.containsKey(name)) {
				continue;
			} else if ((fold == 1) && sourceSubjects.containsKey(name)) {
				continue;
			} else if (entry.isDirectory()) {
				allNames.put(name, new LinkedList<String>());
			} else {
				allNames.get(name).addLast(entry.getName());
			}
		}
		zip.close();

		LinkedList<LinkedList<String>> all = new LinkedList<LinkedList<String>>(
				allNames.values());
		Random rand = new Random(0);
		LinkedList<String> retVal = all.getFirst();
		LinkedList<String> other = all.getLast();
		if (all.getFirst().size() > all.getLast().size()) {
			retVal = all.getLast();
			other = all.getFirst();
		}
		int N = retVal.size();
		for (int i = 0; i < N; ++i) {
			int n = rand.nextInt(other.size());
			String f = other.get(n);
			other.remove(n);
			retVal.add(f);
		}

		return retVal;
	}

	public static void main(String[] args) throws Exception {
		TextExperiment.ZIP_FILE = new File("testData" + File.separator
				+ "sraa_processed.zip");
		for (String[] pair : classes) {
			SRAAexperiment experiment = new SRAAexperiment(pair[0], pair[1]);
			experiment.runExperiment();

		}
	}

}
