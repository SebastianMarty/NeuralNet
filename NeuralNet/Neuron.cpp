#include "Neuron.h"
#include <math.h>

void Neuron::TransferFunction()
{
	_value = 1 / (1 + exp(-_value));
}

double Neuron::TransferFunctionDerivative(double value)
{
	double i = 1 / (1 + exp(-value));
	return i * (1 - i);
}

void Neuron::CalcOutputGradients(double expValue)
{
	double delta = expValue - _value;
	_gradient = delta * TransferFunctionDerivative(_value);
}

void Neuron::CalcHiddenGradients(std::vector<Neuron*> nextLayer)
{
	double dow = SumDow(nextLayer);
	_gradient = dow * TransferFunctionDerivative(_value);
}

double Neuron::SumDow(std::vector<Neuron*> nextLayer)
{
	double sum = 0.0f;

	// sum or contributions of the errors at the nodes we feed

	for (int x = 0; x < nextLayer.size(); x++)
	{
		sum += _weights[x] * nextLayer[x]->_gradient;
	}

	return sum;
}

Neuron::Neuron(double value)
{
	_value = value;
}

void Neuron::Activate()
{
	TransferFunction();
}

void Neuron::AddWeight(double weight)
{
	_weights.push_back(weight);
	_deltaWeights.push_back(0.0f);
}

std::vector<double> Neuron::GetAllWeights()
{
	return _weights;
}

void Neuron::SetWeights(std::vector<double> weights)
{
	_weights = weights;
}

double Neuron::GetValue()
{
	return _value;
}

void Neuron::SetValue(double value)
{
	_value = value;
}

double Neuron::GetGradient()
{
	return _gradient;
}

void Neuron::UpdateWeights(std::vector<Neuron*> prevLayer, int neuronIndex, double learningRate)
{
	// the weights to be updated are in the connection container in the neurons in the preceding layer

	for (int x = 0; x < prevLayer.size(); x++)
	{
		Neuron* prevNeuron = prevLayer[x];

		double oldDeltaWeight = prevNeuron->_deltaWeights[neuronIndex];
		double newDeltaWeight = learningRate * prevNeuron->GetValue() * _gradient + _alpha * oldDeltaWeight;

		prevNeuron->_deltaWeights[neuronIndex] = newDeltaWeight;
		prevNeuron->_weights[neuronIndex] += newDeltaWeight;
	}
}

void Neuron::ClearDeltaWeights()
{
	for (int x = 0; x < _deltaWeights.size(); x++)
	{
		_deltaWeights[x] = 0.0f;
	}
}

double Neuron::GetWeightAt(int index)
{
	return _weights[index];
}