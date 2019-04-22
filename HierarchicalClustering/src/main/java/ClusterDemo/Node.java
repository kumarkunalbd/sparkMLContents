package ClusterDemo;


public class Node {
	Node left, right;
	Node join;
	int id;
	boolean joint = false;
	String printText = "";
	public Node(int id) {
		this.id = id + 1;
		printText = "P" + this.id;
	}
	public Node(Node a, Node b, boolean y) {
		if(y) {
			this.left = a;
			this.right = b;
			this.joint = true;
			this.printText = a.printText + ", " + b.printText;
		}
	}
	public void printJoined() {
		if(!this.joint) {
			System.out.println("This node is not yet grouped into a cluster");
		}else {
			System.out.println("Cluster-- "+this.printText + " are combined");
		}
	}
	
	public Node getRoot(int x, Node root) {
		if(isPresent(x, root)) {
			return root;
		}else {
			return null;
		}
	}
	
	public boolean isPresent(int x, Node parent) {
		if(parent == null) {
			return false;
		}
		if(parent.id == x + 1) {
			return true; 
		}else {
			if(parent.left == null && parent.right == null) {
				return false;
			}else {
				return isPresent(x, parent.left) || isPresent(x, parent.right);
			}
		}
	}
	
}
