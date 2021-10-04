#include "Neuron.h"
#include <vector>
#include <iostream>

using namespace std;

int hiddenLayersCount = 2;
int outputNeuronsCount = 2;
int hiddenNeuronsCount = 10;

double learningRate = 0.1;

vector<double> inputs{ 1.0f, 0.3f, 0.4f, 1.0f, 0.5f };
vector<double> biases;
vector<double> expOutputs = { 1.0f, 0.0f };
vector<vector<Neuron*>> neurons;
vector<Neuron*> inputNeurons;
vector<Neuron*> outputNeurons;

double Randf()
{
	return rand() / double(RAND_MAX);
}

void InitializeNet()
{

	for (double input : inputs)
	{
		inputNeurons.push_back(new Neuron(input));
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
				prevNeuron->AddWeight(Randf());
				value += prevNeuron->GetValue() * prevNeuron->GetWeightAt(y);
			}

			value += biases[x];
			Neuron* neuron = new Neuron(value);
			neuron->Activate();
			hiddenNeurons.push_back(neuron);
		}

		biases.push_back(Randf());

		neurons.push_back(hiddenNeurons);
	}

	for (int x = 0; x < outputNeuronsCount; x++)
	{
		double value = 0.0f;

		for (Neuron* prevNeuron : neurons.back())
		{
			prevNeuron->AddWeight(Randf());
			value += prevNeuron->GetValue() * prevNeuron->GetWeightAt(x);
		}

		value += biases.back();
		Neuron* neuron = new Neuron(value);
		neuron->Activate();
		outputNeurons.push_back(neuron);
	}

	biases.push_back(Randf());

	neurons.push_back(outputNeurons);
}

void BackPropagate()
{
	for (int x = neurons.size() - 1; x >= 0; x--)
	{
		for (int y = 0; y < neurons[x].size(); y++)
		{
			if (x == neurons.size() - 1)
			{
				neurons[x][y]->SetError(expOutputs[y] * neurons[x][y]->SigmoidTransferFunctionDerivative(neurons[x][y]->GetValue()));
			}
			else
			{
				double error = 0.0f;

				for (int z = 0; z < neurons[x + 1].size(); z++)
				{
					error += neurons[x][y]->GetWeightAt(z) * neurons[x + 1][z]->GetError();
					neurons[x][y]->SetError(error * neurons[x][y]->SigmoidTransferFunctionDerivative(neurons[x][y]->GetValue()));
				}
			}
		}
	}
}

void UpdateWeights()
{
	for (int x = 0; x < neurons.size(); x++)
	{
		for (Neuron* neuron : neurons[x])
		{
			neuron->AdjustWeights(learningRate);
		}
	}
}

void ShowAll()
{
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
			cout << "Neuron Value: " << neuron->GetValue() << endl;

			cout << "Neuron Weights:" << endl;

			for (double weight : neuron->GetAllWeights())
			{
				cout << weight << endl;
			}

			cout << "---------------------------" << endl;
			cout << "Neuron Error: " << neuron->GetError() << endl;
			cout << "+++++++++++++++++++++++++++" << endl;
		}
		cout << endl;
		cout << endl;
	}
}

int main()
{
	InitializeNet();

	BackPropagate();

	UpdateWeights();

	ShowAll();
}