#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

void main(int argc, char* argv[]) {

	int world_rank, world_size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	//declaration of the variables
	const int n = 2;
	const int m = (int)(log(n) / log(2));
	int A[n*n], B1[n*n], B[n*n], C[n*n], C1[n], j, i, c[n];
	int a[n], b[n], p = 0;
	int rank1[n], rank2[n], rank3[n*n], rank4[n*n];

	if (world_rank == 0) {
		printf("The value of m is: %d\n", m);
		fflush(stdout);
		printf("Enter the values of A and B matrices:\n");
		fflush(stdout);
		for (i = 0; i < n*n; i++)	scanf("%d", &A[i]);			//take the values of the matrix as an input
		for (i = 0; i < n*n; i++) {
			C[i] = 0;
			scanf("%d", &B1[i]);
		}

		for (i = 0; i < n; i++) {								//transpose the matrix B
			for (j = 0; j < n; j++) {
				B[i*n + j] = B1[i + j*n];
			}
		}

		//initialize the values of the rank matrices
		int k1 = 0, k2 = 0;
		int j;
		for (i = 0; i < n*n; i++) {
			if ((i % (n + 1)) == 0)
				rank1[k1++] = i;
			if ((i % (n)) == 0)
				rank2[k2++] = i;
			rank3[i] = i;
		}

		for (i = 0; i < n; i++) {								//transpose rank3 to get rank4
			for (j = 0; j < n; j++) {
				rank4[i*n + j] = rank3[i + j*n];
			}
		}

	}

	MPI_Bcast(rank1, n, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(rank2, n, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(rank3, n*n, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(rank4, n*n, MPI_INT, 0, MPI_COMM_WORLD);

	memset(a, 0, sizeof(a));
	memset(b, 0, sizeof(b));

	MPI_Group MPI_GROUP_WORLD, g1;
	MPI_Comm s1;
	MPI_Comm_group(MPI_COMM_WORLD, &MPI_GROUP_WORLD);

	//step 1 of the algorithm
	MPI_Group_incl(MPI_GROUP_WORLD, n, rank1, &g1);
	MPI_Comm_create(MPI_COMM_WORLD, g1, &s1);

	if (world_rank % (n+1) == 0) {
		MPI_Scatter(A, n, MPI_INT, a, n, MPI_INT, 0, s1);
	}


	//step 2 of the algorithm
	MPI_Group g2;
	MPI_Comm s2;
	MPI_Group_incl(MPI_GROUP_WORLD, n, rank2, &g2);
	MPI_Comm_create(MPI_COMM_WORLD, g2, &s2);

	if (world_rank % (n) == 0) {
		MPI_Scatter(B, n, MPI_INT, b, n, MPI_INT, 0, s2);
	}

	//step 3 of the algorithm
	MPI_Group g3[n];
	MPI_Comm s3[n];
	
	MPI_Group_incl(MPI_GROUP_WORLD, n, (rank3 + ((world_rank / n)*n)), &g3[world_rank / n]);
	MPI_Comm_create(MPI_COMM_WORLD, g3[world_rank / n], &s3[world_rank / n]);

	
	//note: while B cast in step3, root remains 0

	MPI_Bcast(b, n, MPI_INT, 0, s3[(world_rank / n)]);
	

	//step 4 of the algorithm
	MPI_Group g4[n];
	MPI_Comm s4[n];


	MPI_Group_incl(MPI_GROUP_WORLD, n, rank4 + ((world_rank%n)*n), &g4[world_rank%n]);
	MPI_Comm_create(MPI_COMM_WORLD, g4[world_rank%n], &s4[(world_rank%n)]);
	
	//note: while B cast in step3, root remains 0

	MPI_Bcast(a, n, MPI_INT, (world_rank%n), s4[(world_rank % n)]);

	for (i = 0; i < n; i++) {
		c[i] = a[i] * b[i];
		p += c[i];
	}

	//step 5 of the algorithm
	
	MPI_Gather(&p, 1, MPI_INT, C1, 1, MPI_INT, world_rank % n, s4[world_rank % n]);


	if ((world_rank % (n+1)) == 0)
		MPI_Gather(C1, n, MPI_INT, C, n, MPI_INT, 0, s1);

	if (world_rank == 0) {
		printf("The resultant matrix is:\n");
		fflush(stdout);
		for (i = 0; i < n*n; i++) {
			printf("%d ", C[i]);
			fflush(stdout);
			if ((i + 1) % n == 0) {
				printf("\n");
				fflush(stdout);
			}
		}
	}
	
	if (world_rank % (n + 1) == 0) {
		MPI_Comm_free(&s1);
		MPI_Group_free(&g1);
	}
	if (world_rank % (n) == 0) {
		MPI_Comm_free(&s2);
		MPI_Group_free(&g2);
	}
	MPI_Group_free(&MPI_GROUP_WORLD);
	MPI_Group_free(&g3[world_rank/n]);
	MPI_Comm_free(&s3[world_rank / n]);
	MPI_Group_free(&g4[world_rank%n]);
	MPI_Comm_free(&s4[world_rank%n]);
	MPI_Finalize();

}