#include "planner_neuron.h"

// C++ includes:
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <fstream>

#include "poisson_generator.h"

// Includes from libnestutil:
#include "numerics.h"

// Includes from nestkernel:
#include "event_delivery_manager_impl.h"
#include "exceptions.h"
#include "kernel_manager.h"

// Includes from sli:
#include "dict.h"
#include "dictutils.h"
#include "doubledatum.h"
#include "integerdatum.h"


/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

mynest::planner_neuron::Parameters_::Parameters_()
  : trial_length_( 1000 )
  , target_( 0.0 )
  , prism_deviation_( 0.0 )
  , baseline_rate_( 10.0 )
  , gain_rate_( 1.0 )
{
}

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
mynest::planner_neuron::Parameters_::get( DictionaryDatum& d ) const
{
  def< long >( d, mynames::trial_length, trial_length_ );
  def< double >( d, mynames::target, target_ );
  def< double >( d, mynames::prism_deviation, prism_deviation_ );
  def< double >( d, mynames::baseline_rate, baseline_rate_ );
  def< double >( d, mynames::gain_rate, gain_rate_ );
}

void
mynest::planner_neuron::Parameters_::set( const DictionaryDatum& d )
{
  updateValue< long >( d, mynames::trial_length, trial_length_ );
  if ( trial_length_ <= 0 )
  {
    throw nest::BadProperty( "The trial length cannot be zero or negative." );
  }

  updateValue< double >( d, mynames::target, target_ );
  updateValue< double >( d, mynames::prism_deviation, prism_deviation_ );
  updateValue< double >( d, mynames::baseline_rate, baseline_rate_ );
  updateValue< double >( d, mynames::gain_rate, gain_rate_ );
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node, and destructor
 * ---------------------------------------------------------------- */

mynest::planner_neuron::planner_neuron()
  : DeviceNode()
  , P_()
  , V_()
{
}

mynest::planner_neuron::planner_neuron( const planner_neuron& n )
  : DeviceNode( n )
  , P_( n.P_ )
{
}

/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void
mynest::planner_neuron::init_state_( const Node& proto )
{
  const planner_neuron& pr = downcast< planner_neuron >( proto );

  device_.init_state( pr.device_ );
}


void
mynest::planner_neuron::init_buffers_()
{
  device_.init_buffers();
}

void
mynest::planner_neuron::calibrate()
{
  double rate = P_.baseline_rate_ + P_.gain_rate_ * (P_.target_ + P_.prism_deviation_);
  rate =  std::max(0.0, rate);
  V_.rate_ = rate;

  device_.calibrate();

  // rate_ is in Hz, dt in ms, so we have to convert from s to ms
  V_.poisson_dev_.set_lambda( nest::Time::get_resolution().get_ms() * V_.rate_ * 1e-3 );
}


void
mynest::planner_neuron::update( nest::Time const& T, const long from, const long to )
{
  assert( to >= 0 && ( nest::delay ) from < nest::kernel().connection_manager.get_min_delay() );
  assert( from < to );

  if ( V_.rate_ <= 0 )
  {
    return;
  }

  nest::Time::ms trial_length_ms(P_.trial_length_);
  nest::Time trial_length(trial_length_ms);

  for ( long lag = from; lag < to; ++lag )
  {
    nest::Time now = T + nest::Time::step( lag );

    nest::DSSpikeEvent e;
    nest::kernel().event_delivery_manager.send( *this, e, lag );
    long n_spikes = 0;

    if ( not device_.is_active( now ) )
    {
      continue; // no spike at this lag
    }

    if ( now.get_ms() > P_.trial_length_ )
    {
      // There is no % operator for nest::Time
      nest::Time now_mod_lenght = now;
      while (now_mod_lenght >= trial_length)
      {
        now_mod_lenght = now_mod_lenght - trial_length;
      }

      nest::delay spike_i = now_mod_lenght.get_steps();

      if ( V_.trial_spikes_.count(spike_i) > 0)
      {
        n_spikes = V_.trial_spikes_[spike_i];
      }
    }
    else
    {
      librandom::RngPtr rng = nest::kernel().rng_manager.get_rng( get_thread() );
      n_spikes = V_.poisson_dev_.ldev( rng );

      V_.trial_spikes_[now.get_steps()] = n_spikes;
    }

    if ( n_spikes > 0 ) // we must not send events with multiplicity 0
    {
      e.set_multiplicity( n_spikes );
      e.get_receiver().handle( e );
    }
  }
}

void
mynest::planner_neuron::event_hook( nest::DSSpikeEvent& e )
{
}
