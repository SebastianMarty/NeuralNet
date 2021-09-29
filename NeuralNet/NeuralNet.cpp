#include "Neuron.h"
#include <vector>
#include <iostream>

using namespace std;

int main()
{
	int hiddenLayersCount = 2;
	int outputNeuronsCount = 2;
	int hiddenNeuronsCount = 10;

	vector<double> inputs{ 1.0f, 0.3f, 0.4f, 1.0f, 0.5f };
	vector<vector<Neuron*>> neurons;
	vector<Neuron*> inputNeurons;

	for (double input : inputs)
	{
		Neuron* neuron = new Neuron(input, NeuronType::INPUT);
		inputNeurons.push_back(neuron);
	}

	neurons.push_back(inputNeurons);

	for (int x = 0; x < hiddenLayersCount; x++)
	{
		vector<Neuron*> hiddenNeurons;

		for (int y = 0; y < hiddenNeuronsCount; y++)
		{
			double value = 0.0f;


		}

		neurons.push_back(hiddenNeurons);
	}
}