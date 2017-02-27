/* 
 * This file contains code that is the parallelized version of 
 * Floyd Warshall: All-Pair-Shortest-Path algorithm, written in C using 
 * MPI. Floyd Warshall algorithm basically finds all pair shortest 
 * paths in a weighted graph and use dynamic programming technique to 
 * solve the problem. 
 *
 * Compiling (For GNU and Oracle, please configure in your .cshrc ):
 *  mpicc -o floyd_warshal_mpi floyd_warshal_mpi.c
 *
 * Running:
 *1) mpiexec -np 10 -host pc01 floyd_warshal_mpi [DATA_FILE_NAME]
 * 
 *                  -- or --
 * 
 *2) mpiexec -np 10 --hostfile [name] floyd_warshal_mpi [DATA_FILE_NAME]
 * 
 * Note:
 * .shosts file is given in the source. You can replace it with [name]
 * 
 * File: floyd_warshal_mpi_enhanced.c       Author: Inteasar Haider
 * Date: January 15, 2017                   Version: V1.0
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/* Print matrix after initialization and calculating shortest path
 * 1 = Print matrix
 * 0 = Do not print matrix
 * 
 */
#define PRINT_MATRIX 0 
#define MIN_VERTICES 4    /* Minimun vertices that should be in graph */     
#define DATA_FOLDER  "../test_data/"
#define ROOT 0
#define WALL_TIME 0
#define CPU_TIME 1

const int INFINITY = 1000000;

int read_num_vertices (void);
void read_data_from_file (int global_dist_matrix[], int num_vertices);
void print_distance_adjacency_matrix (int global_dist_matrix[],
				      int num_vertices);
void calculate_all_pair_shortest_path (int local_dist_matrix[],
				       int num_vertices, int my_rank,
				       int num_processes,
				       MPI_Comm comm);
int find_process_rank (int k, int num_processes, int num_vertices);
void copy_row (int local_dist_matrix[], int num_vertices,
	       int num_processes, int row_k[],
	       int k);
void print_row (int local_dist_matrix[], int num_vertices, int my_rank,
		int i);
void master ();
void slave ();
void calculate_time (int my_rank, double my_time, int num_processes,
		     int type_calc);

char final_path[40];  /* Global data file path */

int main (int argc, char* argv[])
{
  int my_rank;
  int num_processes;
  double my_time;    /*variables used for gathering timing statistics*/
  
  if( argc == 2 ) {  /* Second CLI param is the name of file */ 

    strcpy( final_path, DATA_FOLDER ); /* copy data folder path */
    strcat( final_path, argv[1] );     /* create path with file name */
    
  } else {
    printf("\n==> Error: Data file name (e.g: test_data_10.dat) not "
	    "passed from command line \n\n\n");
    return EXIT_FAILURE;
  }

  /* Initialize MPI */
  if (MPI_Init (&argc, &argv) != MPI_SUCCESS)
  {
    printf ("MPI_Init failed.\n");
    MPI_Abort (MPI_COMM_WORLD, EXIT_FAILURE);
    exit (EXIT_FAILURE);
  }

  MPI_Barrier (MPI_COMM_WORLD); /*synchronize all processes*/
  my_time = MPI_Wtime (); /*get time just before work section */

  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &num_processes);

  if (my_rank == ROOT)
  {
    master ();
  }
  else
  {
    slave ();
  }

  my_time = MPI_Wtime () - my_time; /*get time just after work section*/
  calculate_time (my_rank, my_time, num_processes, WALL_TIME);

  MPI_Finalize ();

  return EXIT_SUCCESS;
}

/* This routine will only be executed by process with rank = ROOT or 0.
 * Responsibility of MASTER process is to initialize and setup data.\
 * Once done, it will scatter data of our global matrix in a row blocks 
 * fashion among all processes where each process would work on that 
 * data. Once all processes are done with calculation, MASTER will 
 * gather and consolidate data that will have the shortest paths among
 * all pair of vertices of given graph into one single global matrix.
 *
 * input parameters:	none
 * output parameters:	Global Matrix that contains the shortest path 
 *			among all pair of vertices
 * return value:	none
 * side effects:	might give wrong results if invalid dataset is 
 *			given. Below are the requirements for data:
 *			1) Negative entries are not allowed. 
 *			2) number of vertices should be evenly 
 *			   divisible by number of processes. 
 *			3) Diagonal of matrix should contain only 0. 
 *			4) Use infinity incase there is no path between 
 *			   vertices.
 *
 */
void master ()
{
  MPI_Comm comm; /* Shorthand for communicator */
  int num_processes, my_rank;
  int total_elements;  /* total elements in a matrix. [E.g: n * n] */
  int process_elements; /* elements for each process. [E.g: n*n/p] */
  int num_vertices; /* number of vertices in the given graph*/
  int* local_dist_matrix; /* matrix to hold per process data / row */
  double my_time;  /*variables used for gathering timing statistics*/

  /* matrix that will hold and scatter data read from data file and 
   * later gather all the data from processes 
   */
  int* global_dist_matrix = NULL;

  comm = MPI_COMM_WORLD;
  MPI_Comm_size (comm, &num_processes);
  MPI_Comm_rank (comm, &my_rank);

  /* read number of vertices available in the dataset file */
  num_vertices = read_num_vertices ();

  /* if vertices cannot be evenly divided among all processes */
  if (num_vertices % num_processes != 0)
  {
    printf ("\n==> Error: # of vertices [%d] should be evenly divisible"
	    " by processes [%d] \n", num_vertices, num_processes);
    MPI_Abort (MPI_COMM_WORLD, EXIT_FAILURE);
    exit (EXIT_FAILURE);
  }

  /* broadcast the number of found vertices to all the processes */
  MPI_Bcast (&num_vertices, 1, MPI_INT, ROOT, comm);

  total_elements = num_vertices * num_vertices;
  process_elements = (num_vertices * num_vertices) / num_processes;

  /* allocate memory for local and global matrices */
  global_dist_matrix = malloc (total_elements * sizeof(int));
  local_dist_matrix = malloc (process_elements * sizeof(int));

  /* now read and populate data to global matrix from data file */
  read_data_from_file (global_dist_matrix, num_vertices);

  print_distance_adjacency_matrix (global_dist_matrix, num_vertices);

  /* divide and distribute matrix by block row among all processes */
  MPI_Scatter (global_dist_matrix, process_elements, MPI_INT,
	       local_dist_matrix, process_elements, MPI_INT, ROOT,
	       comm);

  free (global_dist_matrix);
  global_dist_matrix = NULL;

  MPI_Barrier (MPI_COMM_WORLD);	 /*synchronize all processes*/
  my_time = MPI_Wtime ();	 /*get time just before work section */
  
  printf("\n==> Call Floyd Warshall to calculate shortest paths.\n\n");
  /* execute the Floyd Warshall to find shortest path per process */
  calculate_all_pair_shortest_path (local_dist_matrix, num_vertices,
				    my_rank, num_processes, comm);

  my_time = MPI_Wtime () - my_time; /*get time just after work section*/
  calculate_time (my_rank, my_time, num_processes, CPU_TIME);

  /* gather rows from local matrices of each process into global */
  global_dist_matrix = malloc (total_elements * sizeof(int));
  MPI_Gather (local_dist_matrix, process_elements, MPI_INT,
	      global_dist_matrix, process_elements, MPI_INT,
	      ROOT, comm);

  print_distance_adjacency_matrix (global_dist_matrix, num_vertices);

  free (global_dist_matrix);
  free (local_dist_matrix);
}

/* This routine will be executed by all the process except ROOT. Upon 
 * getting the data (in form of row block), this routine will find the 
 * shortest path and update its local matrix. Once all the calculations 
 * are done, it will send back the final data in local matrix to ROOT. 
 *
 * input parameters:	none
 * output parameters:	local matrix contains the shortest distance 
 *			between all pair of vertices
 * return value:	nothing
 * side effects:	no side effects
 *
 */
void slave ()
{
  MPI_Comm comm; /* Shorthand for communicator */
  int num_processes, my_rank;
  int process_elements; /* matrix elements per process. [E.g: n*n/p] */
  int num_vertices; /* number of vertices in the given graph*/
  int* local_dist_matrix;
  double my_time;  /*variables used for gathering timing statistics*/

  int* global_dist_matrix = NULL;

  comm = MPI_COMM_WORLD;
  MPI_Comm_size (comm, &num_processes);
  MPI_Comm_rank (comm, &my_rank);

  /* getting total number of vertices provided in datafile from root*/
  MPI_Bcast (&num_vertices, 1, MPI_INT, ROOT, comm);

  process_elements = (num_vertices * num_vertices) / num_processes;

  local_dist_matrix = malloc (process_elements * sizeof(int));

  /* getting rows of global matrix from root process */
  MPI_Scatter (global_dist_matrix, process_elements, MPI_INT,
	       local_dist_matrix, process_elements, MPI_INT, ROOT,
	       comm);

  MPI_Barrier (MPI_COMM_WORLD);	 /*synchronize all processes*/
  my_time = MPI_Wtime ();	 /*get time just before work section */
  
  /* execute Floyd Warshall to calculate shortest path */
  calculate_all_pair_shortest_path (local_dist_matrix, num_vertices,
				    my_rank, num_processes, comm);

  my_time = MPI_Wtime () - my_time; /*get time just after work section*/
  calculate_time (my_rank, my_time, num_processes, CPU_TIME);

  /* sending final calculated data back to root process */
  MPI_Gather (local_dist_matrix, process_elements, MPI_INT,
	      global_dist_matrix, process_elements, MPI_INT,
	      ROOT, comm);

  free (local_dist_matrix);
}

/* Responsibility of this routine is to read the number of vertices 
 * available in the data-set file
 * 
 * input parameters:	none
 * output:              vertices count of graph given in data file
 * return value:	number of vertices read from the data file
 * side effects:	will abort the whole program if file in not 
 *			found or the number of vertices read from files 
 *			are not enough for further execution
 *
 */
int read_num_vertices (void)
{
  FILE *data_file;
  int num_vertices;

  printf ("\n==> reading number of vertices from file..\n");

  data_file = fopen (final_path, "r");

  if (data_file == NULL)
  {
    printf ("==> Error: Cannot proceed, unable to find data file.\n\n");
    MPI_Abort (MPI_COMM_WORLD, EXIT_FAILURE);
    exit (EXIT_FAILURE);
  }

  /* The first line will be the number of vertices */
  fscanf (data_file, "%d", &num_vertices);

  if (num_vertices <= MIN_VERTICES) /* graph shouldn't be empty */
  {
    printf ("==> Error: Provided number of vertices (%d) are less than"
	    " minimum required vertices (%d).\n\n", num_vertices,
	    MIN_VERTICES);
    MPI_Abort (MPI_COMM_WORLD, EXIT_FAILURE);
    exit (EXIT_FAILURE);
  }

  printf ("\n==> Available vertices in file: %d ...\n", num_vertices);

  return num_vertices;
}

/* Responsibility of this routine is to populate the temporary distance 
 * matrix with the test data being read from file
 * 
 * input parameters:	1) global_dist_matrix: to store date from file
 *			2) num_vertices: vertices needs to be read
 * output:              distance matrix with test data read from file
 * return value:	none 
 * side effects:	abort the whole program if data file is not 
 *			found
 *
 */
void read_data_from_file (int global_dist_matrix[], int num_vertices)
{
  FILE *data_file;
  int temp = 0; /* hold data read from file temporarily */

  data_file = fopen (final_path, "r");

  if (data_file == NULL)
  {
    printf ("\n==> Cannot proceed, unable to find data file ..\n\n");
    MPI_Abort (MPI_COMM_WORLD, EXIT_FAILURE);
    exit (EXIT_FAILURE);
  }

  /* The first line will be the number of vertices */
  fscanf (data_file, "%d", &num_vertices);
  printf ("\n==> Distances between %d vertices is being read ..\n",
	  num_vertices);

  for (int i = 0; i < num_vertices; i++)
  {
    for (int j = 0; j < num_vertices; j++)
    {
      /* reading every integer until end of file */
      if (fscanf (data_file, "%d", &temp) == EOF)
      {
	break;
      }
      else
      {
	if (i == j)
	{
	  /* distance between same vertex should be set zero */
	  global_dist_matrix[i * num_vertices + j] = 0;
	}
	else
	{
	  /* reading and copying data from file. 
	   * If distance between i and j is 0, set it as INFINITY
	   */
	  global_dist_matrix[i * num_vertices + j] =
		  (temp == 0) ? INFINITY : temp;
	}
      }
    }
  }
  fclose (data_file);
  printf ("\n==> Data successfully loaded from file to matrix ..\n\n");
}

/* Responsibility of this routine is to print two dimensional matrix
 * that contains the distance between all pair of vertices of 
 * given graph.
 * 
 * input parameters:	1) distance matrix to print
 *			2) total number of vertices in given matrix
 * output:              distance matrix printed on the console
 * return value:	none
 * side effects:	could crash console if a huge matrix is being 
 *                      printed
 *
 */
void print_distance_adjacency_matrix (int distance_matrix[],
				      int num_vertices)
{
  if (PRINT_MATRIX == 1)
  {
    int i, j;

    printf ("\nMatrix : \n");
    for (i = 0; i < num_vertices; i++)
    {
      for (j = 0; j < num_vertices; j++)
      {
	if (distance_matrix[i * num_vertices + j] == INFINITY)
	{
	  printf ("i ");
	}
	else
	{
	  printf ("%d ", distance_matrix[i * num_vertices + j]);
	}
      }
      printf ("\n");
    }
    printf ("\n\n");
  }
}

/*
 * This routine will execute the Parallel version of Floyd's algorithm 
 * based on a one-dimensional decomposition of the adjacency matrix 
 * to find the shortest path between all pair of vertices.
 * Rows of given adjacency matrix is distributed evenly among all the 
 * processes running.
 * 
 * input parameters:	1) local_mat: data/row of each process.
 *			2) num_vertices: total number of vertices in 
 *					 given graph.
 *			3) my_rank: Rank of a process.
 *			4) num_processes: total number of process 
 *					  executing the program.
 *			5) comm: name of communicator
 * 
 * output:              shortest distance between pair of vertices in 
 *			local matrix of each process
 * return value:	none
 *
 */
void calculate_all_pair_shortest_path (int local_mat[],
				       int num_vertices, int my_rank,
				       int num_processes, MPI_Comm comm)
{
  /* possible shortest path between i and j via intermediate vertex k */
  int poss_pth;

  int global_k, local_i, global_j;
  int root; /* to store process rank that owns a particular row */
  int* row_k = malloc (num_vertices * sizeof(int));

  for (global_k = 0; global_k < num_vertices; global_k++)
  {
    root = find_process_rank (global_k, num_processes, num_vertices);
    if (my_rank == root)
    {
      copy_row (local_mat, num_vertices, num_processes, row_k,
		global_k);
    }

    /* distribute row copied from local matrix among all processes */
    MPI_Bcast (row_k, num_vertices, MPI_INT, root, comm);

    for (local_i = 0; local_i < num_vertices / num_processes; local_i++)
    {
      for (global_j = 0; global_j < num_vertices; global_j++)
      {
	poss_pth = local_mat[local_i * num_vertices + global_k] +
		row_k[global_j];

	if (poss_pth < local_mat[local_i * num_vertices + global_j])
	{
	  local_mat[local_i * num_vertices + global_j] = poss_pth;
	}
      }
    }
  }
  free (row_k);
}

/* Responsibility of this routine is to calculate and return rank 
 * of process that owns row k
 * 
 * input parameters:	1) k: global row number (global_k)
 *			2) num_processes: total number of process
 *			3) num_vertices: total vertices in graph
 * output:              none
 * return value:	rank of the process that owns k (th) row
 * side effects:	none
 *
 */
int find_process_rank (int k, int num_processes, int num_vertices)
{
  return k / (num_vertices / num_processes);
}

/* Responsibility of this routine is to copy the row from the local 
 * matrix of a particular process into row_k array
 * 
 * input parameters:	1) local_dist_matrix: local matrix per process
 *			2) num_vertices: total vertices in graph
 *			3) num_processes: total number of process
 *			4) row_k: location where row needs to be copied
 *			5) k: global row number (global_k)
 * 
 * output:              copy the local_k row of a process in row_k array 
 * return value:	none
 * side effects:	none
 *
 */
void copy_row (int local_dist_matrix[], int num_vertices,
	       int num_processes, int row_k[], int k)
{
  int j;

  /* finding exactly which row of local matrix should be copied */
  int local_k = k % (num_vertices / num_processes);

  for (j = 0; j < num_vertices; j++)
  {
    row_k[j] = local_dist_matrix[local_k * num_vertices + j];
  }
}

/* This function will print a whole row as a string
 * 
 * input parameters:	1) local_dist_matrix: local matrix per process
 *			2) num_vertices: total vertices in graph
 *			3) my_rank: process rank
 *			4) i: row number in local_dist_matrix that 
 *			      needs to be printed
 * 
 * output:              print whole row i of rank my_rank on console 
 * return value:	none
 * side effects:	none
 *
 */
void print_row (int local_dist_matrix[], int num_vertices, int my_rank,
		int i)
{
  char char_int[100];
  char char_row[1000];
  int j, offset = 0;

  for (j = 0; j < num_vertices; j++)
  {
    if (local_dist_matrix[i * num_vertices + j] == INFINITY)
    {
      sprintf (char_int, "i ");
    }
    else
    {
      sprintf(char_int, "%d ", local_dist_matrix[i * num_vertices + j]);
    }
    sprintf (char_row + offset, "%s", char_int);
    offset += strlen (char_int);
  }
  printf ("Proc %d > row %d = %s\n", my_rank, i, char_row);
}

/* This function will print the calculated time on console
 * 
 * input parameters:	1) my_rank: process rank
 *			2) my_time: execution time per process
 *			3) num_processes: total number of processes
 *			4) type_calc: calculating wall or cpu time
 *
 */
void calculate_time (int my_rank, double my_time, int num_processes,
		     int type_calc)
{
  double max_time, /* maximum time taken by a process */
	  min_time, /* minimum time taken by a process */
	  avg_time; /* Average time taken by all processes */

  /*compute max, min, and average timing statistics*/
  MPI_Reduce (&my_time, &max_time, 1, MPI_DOUBLE, MPI_MAX,
	      ROOT, MPI_COMM_WORLD);
  MPI_Reduce (&my_time, &min_time, 1, MPI_DOUBLE, MPI_MIN,
	      ROOT, MPI_COMM_WORLD);
  MPI_Reduce (&my_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM,
	      ROOT, MPI_COMM_WORLD);

  if (my_rank == ROOT)
  {
    avg_time /= num_processes;

    if (type_calc == WALL_TIME)
    {
      printf("\n\nWall time (in seconds) ==> "
	      "Min: %lf  Max: %lf  Avg:  %lf\n\n",
	      min_time, max_time, avg_time);
    }
    else
    {
      printf ("\n\nCPU time (in seconds) ==> "
	      "Min: %lf  Max: %lf  Avg:  %lf\n\n",
	      min_time, max_time, avg_time);
    }
  }
}
