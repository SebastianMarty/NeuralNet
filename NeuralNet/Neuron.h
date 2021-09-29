#pragma once

class Neuron
{
private:
	double _weight = 0.0f;
	double _value = 0.0f;

public:

	Neuron(double value, double weight);

	void Activate();

	void SetWeight(double weight);

	double GetWeight();

	void SetValue(double value);

	double GetValue();

	void Sigmoid();

	double TransferFunctionDerivative(double output);
};