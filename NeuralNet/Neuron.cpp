#include "Neuron.h"
#include <vector>
#include <math.h>

double randf()
{
	return rand() / double(RAND_MAX);
}

double sigmoid(double value)
{
	return value / (1 + abs(value));
}

double transferFunctionDerivative(double output)
{
	return output * (1.0 - output);
}

Neuron::Neuron(double bias = randf(), double weight = randf(), double value, NeuronType type)
{
	_bias = bias;
	_weight = weight;
	_value = value;
	_type = type;
}

void Neuron::Activate()
{
	sigmoid(_value);
}

void Neuron::setBias(double bias)
{
	_bias = bias;
}

double Neuron::getBias()
{
	return _bias;
}

void Neuron::SetWeight(double weight)
{
	_weight = weight;
}

double Neuron::getWeight()
{
	return _weight;
}

void Neuron::setValue(double value)
{
	_value = value;
}

double Neuron::getValue()
{
	return _value;
}