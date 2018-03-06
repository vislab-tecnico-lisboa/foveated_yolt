#ifndef CLASSDATA_H
#define CLASSDATA_H

class ClassData{

public:

    ClassData(int N_): N(N_){  // construtor
      label.resize(N);
      score.resize(N);
      index.resize(N);
    }

    int N;
    std::vector<string> label;
    std::vector<float> score;
    std::vector<int> index;

    friend ostream &operator<<( ostream &output,const ClassData &D ){
        for(int i=0; i<D.N;++i){
            output << " Index: "      << D.index[i] << "\n"
                   << " Label: "      << D.label[i] << "\n"
                   << " Confidence: " << D.score[i] << "\n" << endl;
        }
        return output;
    }
};



#endif // CLASSDATA_H
