#pragma once
#include <vector>

class Neuron
{
private:
	double _value = 0.0f;
	double _gradient = 0.0f;
	double alpha = 0.5f; // [0.0..n] multiplier of last weight change (momentum)

	std::vector<double> _weights;
	std::vector<double> _deltaWeights;

public:

	Neuron(double value);

	void Activate();

	void AddWeight(double weight);

	std::vector<double> GetAllWeights();

	double GetWeightAt(int index);

	void SetWeights(std::vector<double> weights);

	double GetValue();

	void SetValue(double value);

	double GetGradient();

	void TransferFunction();

	void UpdateWeights(std::vector<Neuron*> prevLayer, int neuronIndex, double learningRate);

	double TransferFunctionDerivative(double output);

	void CalcOutputGradients(double expValue);

	void CalcHiddenGradients(std::vector<Neuron*> nextLayer);

	double SumDow(std::vector<Neuron*> nextLayer);
};