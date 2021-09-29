#include "Neuron.h"
#include <vector>
#include <math.h>

void Neuron::Sigmoid()
{
	_value = _value / (1 + abs(_value));
}

double Neuron::TransferFunctionDerivative(double output)
{
	return output * (1.0 - output);
}

Neuron::Neuron(double value, double weight)
{
	_weight = weight;
	_value = value;
}

void Neuron::Activate()
{
	Sigmoid();
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