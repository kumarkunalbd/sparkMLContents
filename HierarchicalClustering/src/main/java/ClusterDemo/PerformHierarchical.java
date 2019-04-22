package ClusterDemo;

public class PerformHierarchical {
	
	static int noOfPoints;
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		// TODO Auto-generated method stub
//		Get data here
		System.out.println("main started");
		double[][] points = new double[][] {
			{1, 1},
			{1.15, 1.15},
			{2, 2},
			{2.02, 2.04},
			{8.8, 8.8},
			{8.9, 8.}
		};
		noOfPoints = points.length;
		DistMatrix distMatrix = new DistMatrix(noOfPoints);
		
		distMatrix.formNewData(points);
		Node[] root = new Node[6];
		int numberOfFreeRoots = 0;
//		need to loop here inorder to reach a single cluster
//		debugging now so no loop
		for(int q = 0; q < 5; q++) {
			int[] minPoints = distMatrix.overallMinimum();
			int firstPoint = Math.min(minPoints[0], minPoints[1]);
			int secondPoint = Math.max(minPoints[0], minPoints[1]);
//			distMatrix.updateDistances(firstPoint, secondPoint);
			int flag = 0;
			Node SecondRoot = null;
			Node FirstRoot = null;
			for(int x = numberOfFreeRoots-1; x >= 0; x--) {
				if(root[x] != null) {
					if(FirstRoot == null && SecondRoot == null) {
						FirstRoot = root[x].getRoot(firstPoint, root[x]);
						SecondRoot = root[x].getRoot(secondPoint, root[x]);
						
					}
					else if(FirstRoot != null && SecondRoot == null) {
						SecondRoot = root[x].getRoot(secondPoint, root[x]);
					}else if(SecondRoot != null && FirstRoot == null) {
						FirstRoot = root[x].getRoot(firstPoint, root[x]);
					}
					
				}
				if(root[x] == null) {
					break;
				}
			}
			
			if(FirstRoot != null & SecondRoot != null) {
				flag = 3;
			}else if(FirstRoot != null) {
				flag = 2;
			}else if(SecondRoot != null) {
				flag = 1;
			}
			
			if(flag == 0) {
				Node leftNode = new Node(firstPoint);
				Node rightNode = new Node(secondPoint);
				root[numberOfFreeRoots] = new Node(leftNode, rightNode, true);
				root[numberOfFreeRoots].printJoined();
				numberOfFreeRoots++;
			}
			else if(flag == 1) {
				Node leftNode = new Node(firstPoint);
				root[numberOfFreeRoots] = new Node(leftNode, SecondRoot, true);
				root[numberOfFreeRoots].printJoined();
				numberOfFreeRoots++;
			}else if(flag == 2) {
				Node rightNode = new Node(secondPoint);
				root[numberOfFreeRoots] = new Node(FirstRoot, rightNode, true);
				root[numberOfFreeRoots].printJoined();
				numberOfFreeRoots++;
			}else if(flag == 3) {
				root[numberOfFreeRoots] = new Node(FirstRoot, SecondRoot, true);
				root[numberOfFreeRoots].printJoined();
				numberOfFreeRoots++;
			}
			
			distMatrix.updateDistances(firstPoint, secondPoint);
		}
				
		
		
	}
	
	

}
