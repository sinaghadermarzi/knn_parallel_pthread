#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include <limits>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

using namespace std;


#define KNN_K 4
double dist(ArffData* dataset,int i,int j)
{
	double sum_sq = 0;
//	std::cout<<i<<","<<j<<":";
	for (int k = 0; k<(dataset->num_attributes()-1);k++)
	{
		double a  = (double) dataset->get_instance(i)->get(k)->operator float();
		double b  = (double) dataset->get_instance(j)->get(k)->operator float();
		sum_sq += (a-b)*(a-b);
//		std::cout<<a<<"-"<<b<<",";
//		std::cin>>a;
	}
	return sqrt(sum_sq);
}



//double dist(ArffData* dataset,int i,int j)
//{
//	return (i-j)*(i-j);
//}


int* KNN(ArffData* dataset)
{
    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));

    for (int i = 0; i < dataset->num_instances(); i++)
    {
    	double * distances = (double*)malloc(dataset->num_instances() * sizeof(double));

    	for (int j = 0; j < (dataset->num_instances()) ; j++)
    	{
    		if (j != i)
    		{
    			distances[j] = dist(dataset, i, j);
    		}
    		else
    		{
    			std::numeric_limits<double>::infinity();
    		}
    	}
//    	for(int k = 0; k<dataset->num_instances();k++)
//    	{
//    		std::cout<<distances[k]<<",";
//
//    	}
//    	int hgk = 2;
//    	std::cin>>hgk;

    	int* votes = (int*) malloc(dataset->num_classes() * sizeof(int) );

    	for(int j =0; j < (dataset->num_classes()) ; j++)
    	{
    		votes [j] =0;
    	}

    	for(int j =0; j < KNN_K ; j++)
    	{
    		int min_cl = 0;//find the class of the jth smallest elment in distances array
    		int min_idx=0;
			double min_dist = std::numeric_limits<double>::infinity();
    		for (int k =0;k<dataset->num_instances();k++)
    		{
    			if ((distances[k]<min_dist)&&(k!=i))
    			{
    				min_dist = distances[k];
    				distances[k] = std::numeric_limits<double>::infinity();
    				min_idx = k;
    				min_cl =dataset->get_instance(k)->get(dataset->num_attributes() - 1)->operator int32();
    			}
    		}
//    		cout<<"min_cl:"<<min_cl<<std::endl;
//    		cout<<"min_dist:"<<min_dist<<std::endl;
//    		cout<<"min_idx:"<<min_idx<<std::endl;
    		votes[min_cl]++;
    	}
    	//find the index of the largest element in votes->prediction
    	int majority_class = 0;
    	int max_votes =0;
//    	int max_votes_idx = 0;

//    	std::cout<<"votes:";
//    	for(int k = 0; k<dataset->num_classes();k++)
//    	{
//    		std::cout<<votes[k]<<",";
//
//    	}
//    	int hgk = 2;
//    	std::cin>>hgk;



    	for (int j = 0;j<(dataset->num_classes());j++)
    	{
    		if (votes[j]>max_votes)
    		{
    			max_votes = votes[j];
    			majority_class=j;
    		}
    	}
    	predictions[i] = majority_class;


    }
//    std::cout<<"predictions:";
//	for(int k = 0; k<dataset->num_instances();k++)
//	{
//		std::cout<<predictions[k]<<",";
//
//	}
//	int hgk = 2;
//	std::cin>>hgk;

    // float attributeValue = dataset->get_instance(instanceIndex)->get(attributeIndex)->operator float();
    // int classValue =  dataset->get_instance(instanceIndex)->get(dataset->num_attributes() - 1)->operator int32();
    
    // Implement KNN here, fill array of class predictions
    
    return predictions;
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matriz size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagnoal are correct predictions
    }
    
    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        cout << "Usage: ./main datasets/datasetFile.arff" << endl;
        exit(0);
    }
    
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    int* predictions = KNN(dataset);
    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    float accuracy = computeAccuracy(confusionMatrix, dataset);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
  
    printf("The KNN classifier for %lu instances required %llu ms CPU time. Accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
}
