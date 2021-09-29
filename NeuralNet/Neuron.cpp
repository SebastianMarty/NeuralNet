#include "Neuron.h"
#include <vector>
#include <math.h>

double Neuron::Randf()
{
	return rand() / double(RAND_MAX);
}

double Neuron::Sigmoid(double value)
{
	return value / (1 + abs(value));
}

double Neuron::TransferFunctionDerivative(double output)
{
	return output * (1.0 - output);
}

Neuron::Neuron(double value, NeuronType type, double bias, double weight)
{
	_bias = bias;
	_weight = weight;
	_value = value;
	_type = type;
}

void Neuron::Activate()
{
	Sigmoid(_value);
}

void Neuron::SetBias(double bias)
{
	_bias = bias;
}

double Neuron::GetBias()
{
	return _bias;
}

void Neuron::SetWeight(double weight)
{
	_weight = weight;
}

double Neuron::GetWeight()
{
	return _weight;
}

void Neuron::SetValue(double value)
{
	_value = value;
}

double Neuron::GetValue()
{
	return _value;
}

NeuronType Neuron::GetType()
{
	return _type;
}
