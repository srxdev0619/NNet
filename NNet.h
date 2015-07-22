#include<armadillo>
#include<string>
#include<iostream>
#include<thread>
//#include<boost/python.hpp>

using namespace std;
using namespace arma;



class NNet
{
 public:
  //Constructor
  NNet();
  //To change the default configuration of the neural Network
  void init(string sconfig, int iclassreg, int inumcores, int igradd, int icostfunc, int iepoch = -1);
  //Stores the activation function of each layer
  void func_arch(string flayer);
  //Load data
  void load(string filename,int imode = 0, string sep1 = ",", string sep2 = " ");
  void test_file(string filename,int verbose = 0,string netname = " ",string sep1 = ",", string sep2 = " ");
  //Train the Nerual Network
  void train_net(double lrate,int mode = 0, int verbose = 0);
  void train_rprop(int mode = 0,int verbose = 0, double tmax = 15.0);
  void d_trainrprop(int mode = 0,int verbose = 0, double tmax = 15.0);
  void test_net(int verbose = 0);
  //Save the current weights and biases
  void savenet(string netname);
  //Load saved net
  void loadnet(string netname);
  //Print saved nets
  void snets(void);
  void ls_init(string nconfig, int iclassreg, int igradd, int icostfunc, int iepoch = -1);
  void ls_load(string ouputfiles, string Qmatrix = " ", string input_file = " ", string sep1 = ",");
  void l_load(string Qmatrix = " ", string input_file = " ", string sep1 = ",");
  void l_init(int numfiles, int iclassreg, int inumcores, int igradd, int icostfunc, int iepoch = -1);
  void l_trainnet(int numlatent, int mode = 0,double tol = -1);
  //void l_testnet(string filename, string netname);
  void l_savenet(void);
  void ls_savenet(string names, string in_name);
  void test_data(string in_filename, string out_filename, string netname, string sep = ",");
  void l_trainrprop(int numlatent,double tmax = 1.0, int mode = 0, double tol = -1);
  void ld_trainrprop(int numlatent, double tmax = 1.0, int mode = 0, double tol = -1);
  void testvoids(int mode);
  void l_funcarch(void);
 private:
  int trained;
  int l_trained;
  //Variables
  // stores the architecture of hidden layers in a array
  //vector<int> paracheck;
  vector<int> config;
  int classreg;
  int numcores;
  int gradd;
  int costfunc;
  int epoch;
  double min_rmse;
  double temp_rmse;
  string loadfile;
  double l_error;
  //Feedforward
  void feed_forward(mat x, int gpos);
  //Backprop
  void backprop(mat x, mat y, int gpos);
  void d_backprop(mat x, mat y, int gpos);
  void parallel_bp(int index, int pos);
  void d_parallelbp(int index, int pos);
  void l_backprop(mat x, mat y, int gpos);
  void l_feedforward(mat x, int gpos);
  void l_parallelbp(int index, int pos);
  void lsavenets(string netname,int index);
  void l_testall(int mode = 0);
  void testfile(int verbose = 0);
  void ld_backprop(mat x, mat y, int gpos);
  void OBD_init(void);
  void l_optimalBD(int pos);
  void ls_optimalBD(void);
  void optimalBD(void);
  void l_update(int r_prop, double r_max);
  void update(int r_prop, double r_max);
  void l_rmsprop(int r_prop);
  void rms_prop(int r_prop);
  vector<mat> params;
  vector<mat> bias;
  vector<mat> velocity;
  vector<mat> best_params;
  vector<mat> best_bias;
  vector<mat> best_velocity;
  //latent parameters algorithim variables
  vector<mat> t_params;
  vector< vector<mat> > l_bias;
  vector< vector<mat> > l_params;
  vector<string> filenames;
  vector< vector<mat> > l_yvals;
  vector<mat> l_xvals;
  vector<int> l_numhids;
  vector< vector<int> > l_numlayers;
  vector< vector<int> > l_funclayer;
  vector< vector<mat> > l_activ;
  vector< vector<mat> > l_sums;
  vector< vector<mat> > l_grads;
  vector< vector<mat> > ld_grads;
  vector< vector<mat> > l_dels;
  vector< vector<mat> > ld_dels;
  vector< vector<mat> > l_tgrads;
  vector< vector<mat> > l_tdels;
  vector< vector<mat> > l_velocity;
  vector< vector<mat> > l_checkgrads;
  vector< vector<mat> > l_checkdels;
  vector< vector<mat> > l_bestparams;
  vector< vector<mat> > l_bestbias;
  vector< vector<mat> > l_saliencies;
  vector< vector<mat> > l_checksals;
  vector< vector<mat> >ld_tgrads;
  vector<mat> ld_tdels;
  vector<mat> ls_saliencies;
  vector<mat> ls_checksals;
  mat lat_checkgrads;
  vector< vector<int> > Q_mat; 
  int file_nlines;
  int l_train;
  int numfiles;
  int l_numx;
  int qmat;
  int l_numlatent;
  //int chk_f;
  //int chk_l;
  //int chk_rw;
  //int chk_cl;
  //Learning rate
  //#//long double alpha;
  //Momentum coeff
  //#//long double nue;
  //Arrays to store training, validation, and test data and thier corresponding numbers
  vector<mat> xdata;
  vector<mat> ydata;
  vector<mat> testxdata;
  vector<mat> testydata;
  int tests;
  int train;
  //int validate;
  int numdata;
  //number of hidden layers
  int numhid;
  //will store length of input vector (pcountx) and output vector(pcounty)
  int pcountx;
  int pcounty;
  //checks wether init has been called or not
  int checkinit;
  //stores the entire configuration of the neural network
  vector<int> numlayers;
  //stores the activation function of each layer
  vector<int> funclayer;
  //Activation functions
  double sigmoid(double x);
  double tanh_net(double x);
  double reclinear(double x);
  double tanh_r(double x);
  double tanh_d(double x);
  double tanh_dr(double x);
  double softplus(double x);
  double rec_D(double x);
  double softplus_D(double x);
  double tanh_dd(double x);
  double sigmoid_dd(double x);
  double sigmoid_d(double x);
  //Stores files
 
  int loadmode;
  vector<mat> tgrads;
  vector<mat> tdels;
  vector<mat> d_tgrads;
  vector<mat> saliencies;
  vector<mat> checksals;
  vector< vector<mat> > activ;
  vector< vector<mat> > sums;
  vector< vector<mat> > grads;
  vector< vector<mat> > dels;
  vector< vector<mat> > d_grads;
  vector< vector<mat> > d_dels;
  vector<mat> checkgrads;
  vector<mat> checkdels;
  //vector<std::thread> bpthreads;
}; 

//0:sigmoid, 1:tanh, 2:reclinear, 3:tanh + 0.1x




