# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - YYYY-MM-DD
### Added
- Add the AI docs, describing the training, data and algorithm evaluation [@wujuu, @JanKapala, @Michal-Kolomanski]
- Flask CLI commands (testing db seed, execution of individual AI pipelines and more) [@wujuu]
- Unit and integration tests for each pipeline step of the autoencoders [@Michal-Kolomanski]
- New ENV variable, which points to default recommendation algorithm [@Michal-Kolomanski]
- Methods refactoring in the AEDataPreparationStep [@Michal-Kolomanski]
- The ability to modify the sizes and number of layers in the autoencoders networks.[@Michal-Kolomanski]
- Refactoring neural network layer definitions across all pipelines [@Michal-Kolomanski]
- Implement migration process [@wujuu]
- Split validation in autoencoder's data preparation step [@Michal-Kolomanski]
- Logging system [@Michal-Kolomanski]  
- Bug fix: flask seed_faker command on an empty database [@Michal-Kolomanski]
- Fix bug: Flask db seed [@Michal-Kolomanski]
- Add MongoDB to CI and to testing environment [@wujuu]

### Changed

### Fixed

### Removed

## [3.0.1] - 2021-11-05
### Added

### Changed
- Training time significantly decreased, by reducing epochs and unused models [@wujuu]

### Fixed
- Fixed RLInference bugs on user's history longer than max history length [@wujuu] 

### Removed

## [3.0.0] - 2021-10-27
### Added
- fix AGENT_VERSION default in execute_training task [@JanKapala]
- Add proper policy delay to TD3 [@wujuu]
- Complex and simple reward generation [@wujuu]
- Add ENGINE_VERSION field to the /recommendations endpoint [@Michal-Kolomanski]
- Abstract Pipeline architecture [@JanKapala, @wujuu]
- NCFPipeline [@JanKapala]
- NCFInferenceComponent [@JanKapala]
- Pipeline and Step Metadata [@wujuu, @JanKapala]
- AutoencodersPipeline [@Michal-Kolomanski]
- Embedder and Normalizers implementation and integration [@JanKapala, @wujuu]
- RLPipeline [@wujuu]

### Changed
- Split History Embedder into MLP and LSTM versions [@wujuu]
- Service engagement refactoring [@wujuu]
- Service Selector refactoring [@wujuu]
- Improved SARSes generator [@JanKapala]
- Project structure [@Michal-Kolomanski, @wujuu, @JanKapala]

### Fixed
- Fix dynamic examples in swagger API [@JanKapala]

### Removed
- Deletion of unnecessary jupyter notebooks [@JanKapala]

## [2.2.2] - 2021-09-20
### Fixed
- fix mongo FTS by specifying proper fields in the service model [@JanKapala]

## [2.2.1] - 2021-07-27
### Fixed
- AGENT_VER defaults to pre_agent if not in env [@JanKapala]

## [2.2.0] 2021-07-22
### Added
- Services History Generator [@JanKapala]
- Base Agent Recommender [@JanKapala]
- RL Agent Recommender [@JanKapala]
- History Embedder [@wujuu]
- Actor Model [@wujuu]
- Synthetic Dataset [@wujuu]
- Service Selector  [@wujuu]
- Services2weights [@wujuu]
- Action Encoder, Action Embedder and Critic [@JanKapala]
- Replay Buffer [@JanKapala]
- Update endpoint, model reloading [@JanKapala]
- Batch compatibility [@JanKapala]
- Continuous Integration [@wujuu]

### Changed
- Represent action as weights [@wujuu]
- Smart Examples in Swagger [@JanKapala]

### Fixed
- Handle invalid /recommendations and /user_actions requests [@JanKapala, @wujuu]

## [2.1.2] - 2021-05-21
### Added
- Services History Generator [@JanKapala]
- Base Agent Recommender [@JanKapala]
- RL Agent Recommender [@JanKapala]
- History Embedder [@wujuu, @JanKapala]
- Actor Model [@wujuu, @JanKapala]
- User Interest Evaluation [@wujuu]
- Reward mapping [@wujuu]
- Action Selector [@wujuu]

### Changed

### Fixed

### Removed

## [2.1.1] - 2021-05-12
### Fixed
- Fixed bugs associated with recommendation of services with specific status (published, unverified). [@JanKapala]

## [2.1.0] - 2021-04-12
### Added
- Fully Operational Pre Agent [@JanKapala]

## [2.0.0] - 2021-04-09
### Changed
- stable API [@JanKapala]

### Fixed
- remove logged_user from tests and fixtures [@JanKapala]

## [1.0.0] - 2021-03-31
### Added
- Flask integration [@JanKapala]
- MongoDB integration [@wujuu]
- Celery integration [@wujuu]
- SARSes generator [@JanKapala]
- Docker integration [@wujuu]
- Neural Collaborative Filtering [@JanKapala]
- User Actions endpoint [@JanKapala, @wujuu]
- Mongo FTS integration [@wujuu]
- Sentry integration [@JanKapala]
- Test Coverage [@JanKapala]