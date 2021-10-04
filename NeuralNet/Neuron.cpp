#include "Neuron.h"
#include <math.h>

void Neuron::Sigmoid()
{
	_value = _value / (1 + abs(_value));
}

double Neuron::SigmoidTransferFunctionDerivative(double output)
{
	return output * (1.0 - output);
}

Neuron::Neuron(double value)
{
	_value = value;
}

void Neuron::Activate()
{
	Sigmoid();
}

void Neuron::AddWeight(double weight)
{
	_weights.push_back(weight);
}

std::vector<double> Neuron::GetAllWeights()
{
	return _weights;
}

void Neuron::SetWeights(std::vector<double> weights)
{
	_weights = weights;
}

void Neuron::SetValue(double value)
{
	_value = value;
}

double Neuron::GetError()
{
	return _error;
}

void Neuron::SetError(double error)
{
	_error = error;
}

void Neuron::AdjustWeights(double learningRate)
{
	for (int x = 0; x < _weights.size(); x++)
	{
		_weights[x] = _weights[x] + learningRate * _error * _input;
	}
}

double Neuron::GetWeightAt(int index)
{
	return _weights[index];
}

double Neuron::GetValue()
{
	return _value;
}