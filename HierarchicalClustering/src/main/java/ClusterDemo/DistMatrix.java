package ClusterDemo;

import java.util.*;

public class DistMatrix {
	List<ArrayList<Double>> distRows;
	int matrixLength = 0;
	int noOfRows = 0;
	public DistMatrix(int l) {

		matrixLength = l;
		noOfRows = l;
		distRows = new ArrayList<ArrayList<Double>>(l);
	}

	
	public void formNewData(double[][] points) {
		for(int i = 0; i < points.length; i++) {
			ArrayList<Double> list = new ArrayList<Double>();
			for(int j = 0; j < i + 1 ; j++) {
				list.add(euclideanDistance(points[i], points[j]));
				distRows.add(i, list);
//				System.out.print(distRows.get(i).get(j) + " ");
			}
		}
	}
		
	public int[] overallMinimum() {
		double minimum = Integer.MAX_VALUE;
		int[] minPoints = {0,0};
		for(int i = 0; i < noOfRows; i++) {
			for(int j = 0; j < i; j++) {
				if((i != j) && distRows.get(i).get(j) < minimum) {
					minimum = distRows.get(i).get(j);
					minPoints[0] = i;
					minPoints[1] = j;
				}
			}
		}		
		System.out.println("P" + (minPoints[0] + 1) + ", P" + (minPoints[1] + 1) + " are nearest");
		return minPoints;
	}
	
	
//	
	public void updateDistances(int p1, int p2) {
		for(int i = 0; i < noOfRows; i++) {
//			System.out.println(i + " " + p1 + " " + p2);
			for(int j = 0; j < i; j++) {

				if(distRows.get(i).get(0) == 10000) {
					break;
				}
				if(i == p1 ) {
					distRows.get(i).set(j, Math.min(distRows.get(p1).get(j), distRows.get(p2).get(j)));
//					System.out.print(distRows.get(i).get(j) + " ");
				}else {
					if(j == p1 && (p2 > i)) {
						
						distRows.get(i).set(j, Math.min(distRows.get(i).get(p1), distRows.get(p2).get(i)));
					}
					if(j == p1 && (p2 < i)) {
						distRows.get(i).set(j, Math.min(distRows.get(i).get(p1), distRows.get(i).get(p2)));
					}
//					System.out.print(distRows.get(i).get(j) + " ");
					continue;
				}
			}
//			System.out.println(" ");
			
		}
//		System.out.println("Here is the size of the masked array " + distRows.get(p2).size());
		for(int i = 0; i < distRows.get(p2).size(); i++) {
			distRows.get(p2).set(i, 10000.00);
			
		}
//		System.out.println(distRows.get(p2));
//		distRows.remove(p2);
//		noOfRows--;
	}
	
	public double euclideanDistance(double[] p1, double[] p2) {
		double dist = 0;
		int len = p1.length;
		for(int i = 0; i < len; i++) {
			dist = dist + Math.pow((p1[i] - p2[i]), 2);
		}
		return Math.sqrt(dist);
	}
	
}
