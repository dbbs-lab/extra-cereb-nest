#ifndef PLANNER_NEURON_H
#define PLANNER_NEURON_H

#include <fstream>
#include <map>

// Includes from librandom:
#include "poisson_randomdev.h"

// Includes from nestkernel:
#include "connection.h"
#include "device_node.h"
#include "event.h"
#include "nest_types.h"
#include "stimulating_device.h"

#include "mynames.h"


namespace mynest
{
class planner_neuron : public nest::DeviceNode
{

public:
  planner_neuron();
  planner_neuron( const planner_neuron& );

  bool
  has_proxies() const
  {
    return false;
  }

  using nest::Node::event_hook;

  nest::port send_test_event( nest::Node&, nest::rport, nest::synindex, bool );

  void get_status( DictionaryDatum& ) const;
  void set_status( const DictionaryDatum& );

private:
  void init_state_( const nest::Node& proto );
  void init_buffers_();
  void calibrate();

  void update( nest::Time const&, const long, const long );
  void event_hook( nest::DSSpikeEvent& );

  struct Parameters_
  {
    long trial_length_;
    double target_;
    double prism_deviation_;
    double baseline_rate_;
    double gain_rate_;

    Parameters_(); //!< Sets default parameter values

    void get( DictionaryDatum& ) const; //!< Store current values in dictionary
    void set( const DictionaryDatum& ); //!< Set values from dicitonary
  };

  struct Variables_
  {
    double rate_; //!< process rate in Hz
    librandom::PoissonRandomDev poisson_dev_; //!< Random deviate generator
    std::map<nest::delay, long> trial_spikes_;
  };

  nest::StimulatingDevice< nest::SpikeEvent > device_;
  Parameters_ P_;
  Variables_ V_;
};

inline nest::port
planner_neuron::send_test_event( nest::Node& target, nest::rport receptor_type, nest::synindex syn_id, bool dummy_target )
{
  device_.enforce_single_syn_type( syn_id );

  if ( dummy_target )
  {
    nest::DSSpikeEvent e;
    e.set_sender( *this );
    return target.handles_test_event( e, receptor_type );
  }
  else
  {
    nest::SpikeEvent e;
    e.set_sender( *this );
    return target.handles_test_event( e, receptor_type );
  }
}

inline void
planner_neuron::get_status( DictionaryDatum& d ) const
{
  P_.get( d );
  device_.get_status( d );
}

inline void
planner_neuron::set_status( const DictionaryDatum& d )
{
  Parameters_ ptmp = P_; // temporary copy in case of errors
  ptmp.set( d );         // throws if BadProperty

  // We now know that (ptmp, stmp) are consistent. We do not
  // write them back to (P_, S_) before we are also sure that
  // the properties to be set in the parent class are internally
  // consistent.
  //nest::Archiving_Node::set_status( d );

  // if we get here, temporaries contain consistent set of properties
  P_ = ptmp;
}

} // namespace

#endif // PLANNER_NEURON_H
