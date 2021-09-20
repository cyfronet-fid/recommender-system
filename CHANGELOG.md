# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - YYYY-MM-DD
### Added

### Changed

### Fixed

### Removed

## [2.2.2] - 2021-09-20
### Fixed
- fix mongo FTS by specifying proper fields in the service model [@JanKapala]

## [2.2.1] - 2021-07-27
### Fixed
- fix AGENT_VERSION default in execute_training task [@JanKapala]

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
- fix AGENT_VERSION default in execute_training task [@JanKapala]

### Removed