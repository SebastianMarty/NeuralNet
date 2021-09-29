#include "Neuron.h"
#include <vector>
#include <iostream>

using namespace std;

double Randf()
{
	return rand() / double(RAND_MAX);
}

int main()
{
	int hiddenLayersCount = 2;
	int outputNeuronsCount = 2;
	int hiddenNeuronsCount = 10;

	vector<double> inputs{ 1.0f, 0.3f, 0.4f, 1.0f, 0.5f };
	vector<double> biases;
	vector<vector<Neuron*>> neurons;
	vector<Neuron*> inputNeurons;
	vector<Neuron*> outputNeurons;

	for (double input : inputs)
	{
		inputNeurons.push_back(new Neuron(input, Randf()));
	}

	biases.push_back(Randf());

	neurons.push_back(inputNeurons);

	for (int x = 0; x < hiddenLayersCount; x++)
	{
		vector<Neuron*> hiddenNeurons;

		for (int y = 0; y < hiddenNeuronsCount; y++)
		{
			double value = 0.0f;

			for (Neuron* prevNeuron : neurons[x])
			{
				value += prevNeuron->GetValue() * prevNeuron->GetWeight();
			}

			value += biases[x];
			Neuron* neuron = new Neuron(value, Randf());
			neuron->Activate();
			hiddenNeurons.push_back(neuron);
		}

		biases.push_back(Randf());

		neurons.push_back(hiddenNeurons);
	}

	for (int x = 0; x < outputNeuronsCount; x++)
	{
		double value = 0.0f;

		for (Neuron* prevNeuron : neurons[x])
		{
			value += prevNeuron->GetValue() * prevNeuron->GetWeight();
		}

		value += biases.back();
		Neuron* neuron = new Neuron(value, Randf());
		neuron->Activate();
		outputNeurons.push_back(neuron);
	}

	biases.push_back(Randf());

	neurons.push_back(outputNeurons);

	for (vector<Neuron*> layer : neurons)
	{
		auto it = find(neurons.begin(), neurons.end(), layer);
		int index = it - neurons.begin();

		cout << "Layer: " << index + 1 << endl;
		cout << "Layer Bias: ";
		cout << biases[index] << endl;
		cout << "***************************" << endl;

		for (Neuron* neuron : layer)
		{
			cout << "Neuron Value: ";
			cout << neuron->GetValue() << endl;

			cout << "Neuron Weight: ";
			cout << neuron->GetWeight() << endl;

			cout << "---------------------------" << endl;
		}

		cout << endl;
		cout << endl;
	}
}