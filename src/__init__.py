import jax

# Hot-path noise (Langevin, read, write) uses the fast "rbg" PRNG. rbg streams
# are independent across runs but depend on the run's position in a vmap batch,
# so identities that must be a pure function of a seed (device draws, weight
# initialisations) use explicit threefry keys: see device.devices_from_seeds
# and eqprop.init_from_seeds.
jax.config.update("jax_default_prng_impl", "rbg")
jax.config.update("jax_enable_x64", False)
