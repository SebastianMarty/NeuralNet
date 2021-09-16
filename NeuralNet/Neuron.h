#pragma once

enum class NeuronType
{
	INPUT,
	HIDDEN,
	OUTPUT
};

class Neuron
{
private:
	double _bias = 0.0f;
	double _weight = 0.0f;
	double _value = 0.0f;

	NeuronType _type;

public:
	Neuron(double bias = randf(), double weight = randf(), double value, NeuronType type);

	void Activate();

	void setBias(double bias);

	double getBias();

	void SetWeight(double weight);

	double getWeight();

	void setValue(double value);

	double getValue();
};