#pragma once
#include <vector>

class Neuron
{
private:
	double _value = 0.0f;
	std::vector<double> _weights;

public:

	Neuron(double value);

	void Activate();

	void AddWeight(double weight);

	void SetWeights(std::vector<double> weights);

	std::vector<double> GetAllWeights();

	double GetWeightAt(int index);

	void SetValue(double value);

	double GetValue();

	void Sigmoid();

	double TransferFunctionDerivative(double output);
};