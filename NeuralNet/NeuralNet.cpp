#include "Neuron.h"
#include <vector>
#include <iostream>

using namespace std;

int hiddenLayersCount = 2;
int outputNeuronsCount = 2;
int hiddenNeuronsCount = 10;
int trainingRounds = 10000;

double learningRate = 0.01;
double avgError = 0.0f;
double recentAvgError = 0.0f;
double recentAvgSmoothingFactor = 100.0f;

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

int GetNeuronIndex(Neuron* neuron, int layerIndex)
{
	auto it = find(neurons[layerIndex].begin(), neurons[layerIndex].end(), neuron);
	return it - neurons[layerIndex].begin();
}

int GetLayerIndex(vector<Neuron*> layer)
{
	auto it = find(neurons.begin(), neurons.end(), layer);
	return it - neurons.begin();
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

void FeedForward()
{
	for (int x = 1; x < neurons.size(); x++)
	{
		for (int y = 0; y < neurons[x].size(); y++)
		{
			double value = 0.0f;

			for (Neuron* prevNeuron : neurons[x - 1])
			{
				value += prevNeuron->GetWeightAt(y) * prevNeuron->GetValue();
			}

			value += biases[x];

			neurons[x][y]->SetValue(value);
			neurons[x][y]->Activate();
		}
	}
}

void BackPropagate()
{
	vector<Neuron*> outputLayer = neurons.back();
	avgError = 0.0f;

	for (int x = 0; x < outputLayer.size(); x++)
	{
		double delta = expOutputs[x] - outputLayer[x]->GetValue();
		avgError += delta * delta;
	}

	avgError /= outputLayer.size(); // get average error squared
	avgError = sqrt(avgError); // root mean squared error (rmse)

	// recent average measurement
	recentAvgError = (recentAvgError * recentAvgSmoothingFactor + avgError) / (recentAvgSmoothingFactor + 1.0f);

	// calculate output layer gradients
	for (int x = 0; x < outputLayer.size(); x++)
	{
		outputLayer[x]->CalcOutputGradients(expOutputs[x]);
	}

	// calculate gradients on hidden layers
	for (int x = neurons.size() - 2; x >= 0; x--)
	{
		vector<Neuron*> hiddenLayer = neurons[x];
		vector<Neuron*> nextLayer = neurons[x + 1];

		for (int y = 0; y < hiddenLayer.size(); y++)
		{
			hiddenLayer[y]->CalcHiddenGradients(nextLayer);
		}
	}

	// for all layers from outputs to first hidden layer, update connection weights
	for (int x = neurons.size() - 1; x > 0; x--)
	{
		vector<Neuron*> layer = neurons[x];
		vector<Neuron*> prevLayer = neurons[x - 1];

		for (int y = 0; y < layer.size() - 1; y++)
		{
			int layerIndex = GetLayerIndex(layer);
			int neuronIndex = GetNeuronIndex(layer[y], layerIndex);
			layer[y]->UpdateWeights(prevLayer, neuronIndex, learningRate);
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
			cout << "Gradient: " << neuron->GetGradient() << endl;
			cout << "Average Error: " << avgError << endl;
			cout << "Recent Average Error: " << recentAvgError << endl;
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

	// Training the network
	for (int x = 0; x < trainingRounds; x++)
	{
		FeedForward();

		BackPropagate();
	}

	ShowAll();
}