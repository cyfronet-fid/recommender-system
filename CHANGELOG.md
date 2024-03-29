# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - YYYY-MM-DD
### Added
- RandomRanking engine [@Michal-Kolomanski]
### Changed
### Fixed
### Removed

## [6.0.2] - 2023-02-28
### Added
### Changed
- RL training is omitted during training [@Michal-Kolomanski]
### Fixed
### Removed

## [6.0.1] - 2023-02-27
### Added
### Changed
### Fixed
- Handling nullable string as service pid [@Michal-Kolomanski]
### Removed

## [6.0.0] - 2023-02-13
### Added
- BREAKING CHANGE: `research_steps` and `horizontal` [@Michal-Kolomanski]
- BREAKING CHANGE: providers `PID` [@Michal-Kolomanski]
### Changed
- `engine_version` optional [@Michal-Kolomanski]
- `panel_id` optional [@Michal-Kolomanski]
- default recommendation engine is NCF [@Michal-Kolomanski]
- `panel_id=service` handled [@Michal-Kolomanski]
### Fixed
### Removed

## [5.0.1] - 2023-01-19
### Added
- /health endpoint [@wujuu]
- pid for services [@wujuu]
### Changed
- **BREAKING CHANGE**: Adjust user actions handling to the new user action schema [@JanKapala]
### Fixed
- engine_version in the recommendations response [@Michal-Kolomanski]
- better exception handling for finding user in the DB [@Michal-Kolomanski]
### Removed

## [4.0.0] - 2022-10-26
### Added
- Implement basic recommendations explanations handling [@JanKapala]
### Changed
- `/recommendations` endpoint for the purposes of the User Dashboard [@Michal-Kolomanski]
- Rename elastic_services to candidates [@JanKapala]
- Adapt Inference Components to return probability scores [@JanKapala]
- **BREAKING CHANGE**: Extend `/recommendations` endpoint response with additional fields [@JanKapala]
### Fixed
### Removed

## [Unreleased] - 2022-06-23
### Added
- Random recommendations' engine for anonymous users [@Michal-Kolomanski]
- Add NCF Ranking Inference Component [@JanKapala]
- 
### Changed
- Cleaning up the main conftest [@Michal-Kolomanski]
- More metadata in autoencoders pipeline [@Michal-Kolomanski]
- Pandas warnings in the AE pipline [@Michal-Kolomanski]
- db commands divided into db and seed commands [@Michal-Kolomanski]
- seed commands only available in development and testing env [@Michal-Kolomanski]
### Fixed
- Dependabot alerts [@Michal-Kolomanski]
### Removed

## [3.2.1] - 2022-05-05
### Added
### Changed
- Python version: 3.10 [@Michal-Kolomanski]
- Updating and cleaning up packages [@Michal-Kolomanski]
### Fixed
- Dependabot alerts [@Michal-Kolomanski]
### Removed

## [3.2.0] - 2022-04-26
### Added
- Integrate elastic_search results from the Marketplace [@Michal-Kolomanski]
- JMS Subscriber for user actions and flask cli for it [@michal-szostak]
- search_data migration [@Michal-Kolomanski]
- tqdm and logging in embedding pipeline [@Michal-Kolomanski]
### Changed
- engine_version field in recommendation request is required [@Michal-Kolomanski]
- Structure of the tests in the tests folder [@Michal-Kolomanski]
- There is no version specified in flask command for RL training. Use: flask train rl [@Michal-Kolomanski]
- drop_ml_models in flask train all [@Michal-Kolomanski]
### Fixed
### Removed
- v2 configuration of RL pipeline removed [@Michal-Kolomanski]

## [3.1.2] - 2022-03-02
### Added
- JMS connection [@Michal-Kolomanski]
- Models from m_l_component collection are deleted when necessary (/update, flask command) [@Michal-Kolomanski]
- Flask command to delete models from m_l_component collection [@Michal-Kolomanski]
- Flask train, db command and docs in the README [@Michal-Kolomanski]
- If requested recommendation model (in /recommendation body request) does not exist try to use another model [@Michal-Kolomanski]
- engine_version in recommendation collection [@Michal-Kolomanski]
### Changed
- Black version = 20.8b0 - click dependency [@Michal-Kolomanski]
### Fixed
- InsufficientRecommendationSpace in RL model evaluation step [@Michal-Kolomanski]
### Removed
- RL v2 training in flask train all [@Michal-Kolomanski]

## [3.1.1] - 2022-01-31
### Added
- User actions documentation [@JanKapala, @Michal-Kolomanski]
- Logging in generation of SARSes [@Michal-Kolomanski]
- SARS generator - saving users with ID -1 to the database while they are not logged in [@Michal-Kolomanski]
### Changed
- Limited SARSes deletion in the RL data extraction step [@Michal-Kolomanski]
- Test for regenerate SARSes are disabled due to the fact that they tend to freeze [@Michal-Kolomanski]
### Fixed
- Details in the AE data preparation step [@Michal-Kolomanski]
- Multiprocessing in Celery task. Transition to billiard [@JanKapala, @Michal-Kolomanski]
### Removed
- Invalid user actions progress bar [@Michal-Kolomanski]

## [3.1.0] - 2022-01-17
### Added
- Add the AI docs, describing the training, data and algorithm evaluation [@wujuu, @JanKapala, @Michal-Kolomanski]
- Flask CLI commands (testing db seed, execution of individual AI pipelines and more) [@wujuu]
- Unit and integration tests for each pipeline step of the autoencoders [@Michal-Kolomanski]
- New ENV variable, which points to default recommendation algorithm [@Michal-Kolomanski]
- The ability to modify the sizes and number of layers in the autoencoders networks.[@Michal-Kolomanski]
- Implement migration process [@wujuu]
- Split validation in autoencoder's data preparation step [@Michal-Kolomanski]
- Logging system [@Michal-Kolomanski]
- Add MongoDB to CI and to testing environment [@wujuu]
- Add adjustable min/max numerical boundaries on Actor output [@wujuu]
- Add more tests to RLInferenceComponent [@wujuu]
- DB indexes [@JanKapala]
- Support for multiprocessing-capable code testing and parallel tests [@JanKapala]
- Implement user journey creator for testing purposes [@JanKapala]
- User user journey visualizer [@JanKapala]
- Add pytest-randomly [@JanKapala]

### Changed
- Methods refactoring in the AEDataPreparationStep [@Michal-Kolomanski]
- Refactoring neural network layer definitions across all pipelines [@Michal-Kolomanski]
- Reimplementation SARSes (re)generation [@JanKapala]
### Fixed
- Flask seed_faker command on an empty database [@Michal-Kolomanski]
- Flask db seed command [@Michal-Kolomanski]
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