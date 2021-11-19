# Introduction
This document contains the overview of the data from which our recommendation algorithms learn, in the context of recommending scientific resources to users of the EOSC Marketplace & Catalogue (MP).

## Users
We are extracting all the relevant information about the user from the MP's database, that is:
- Categories - categories that users have defined as being of interest
- Scientific domains - scientific domains which users labelled as interesting to them
- Accessed services - ids of the services added by the users to their projects, ordered chronologically 

User data is later encoded to one-hot vector by the transformer:
<p align="center">
  <img src="https://user-images.githubusercontent.com/30239467/142583205-9c183bac-9290-47da-b245-9da60f369683.png" alt="drawing" width="500"/>
</p>

## Services
The same is true for services, which we will be recommending to the users. Information extracted from the MP's database is as follows:
- Categories - categories assigned to the given service by its provider
- Scientific domains - scientific domains assigned to the given service by its provider
- Geographical locations - locations in which the service is available
- Order type - specifies whether the service is open-access or requires an order
- Providers - list of providers that are supplying the service
- Related platforms - platforms on which the service is available
- Target users - specifies the target group of the service (e.g. researchers, students)

Similarly to user data, the service data is encoded to one-hot vectors by the transformer:
<p align="center">
  <<img src="https://user-images.githubusercontent.com/30239467/142583388-07a25e37-655e-4c2d-bb38-1dacdbeee942.png" alt="drawing" width="800"/>
</p>

## Autoencoders
We are using AutoEncoders to achieve a better, more dense vector representation of the mentioned one-hot encodings. AutoEncoders are trained to encode and then decode the vectors as closely to the original as possible. Following training, the "decoder" component is utilized to embed the vectors in a more desirable space.

Embedded user and service vectors are then used as direct input to the NCF and the RL algorithms (while the RL also requires that we encode some additional data - explained in the next section)

## TD3 specific data
All Reinforcement Learning algorithms need to be given a specific state, action and reward. This section will describe all the required shapes in detail.

### State (S)
It contains the recommendation context:
- All the user data, as described in [users](#users)
- History - all of the `accessed_services` as described in [users](#users) in addition to all of the services that the user has shown interest in (not just the ones that they added to their projects). The interest means clicking on the service displayed in the recommendation panel, going to its details (opinions, details, related external websites), etc. The highest form of user’s interest is to order a given service. 
- Filters and search phrase. The recommendations panel is presented on the page with the list of services. Since the most recent user preferences should be taken into account when calculating the recommendations, the search parameters are also taken into consideration, i.e. the search phrase as well as the selected filters. The filters include the following items (represented by the list of ids):
  - categories
  - geographical availabilities
  - order type
  - providers
  - related platforms
  - scientific domains
  - target users
 
The state information is then taken through various encoders and embedders. User data is embedded exactly as described in [users](#users). Each service is also embedded exactly as described in [services](#services) and then concatenated. The search data is encoded by the search data encoder as follows:
<p align="center">
  <<img src="https://user-images.githubusercontent.com/30239467/142587380-7d74a837-2831-486e-806d-11fb25402d39.png" alt="drawing" width="600"/>
</p>

The big picutre of state encoding:
<p align="center">
  <<img src="https://user-images.githubusercontent.com/30239467/142471449-33208327-ef01-48b8-9aad-eb09e5ebee3e.png" alt="drawing" width="600"/>
</p>

### Action (A)
An action is a tuple of K services that are presented in the MP's recommendation panel as recommendations, where K = {2, 3} in the current version of the system, but can be treated as a constant.

```
A = (service id 1 , service id 2 , ..., service id K)
```
Each of the action service is embedded as in [service](#services). 

### Reward (R)
After the recommendation (action) is sent to the Marketplace, the user can click (or ignore) each of the K recommended services, explore the pages of each of the services and even order them. The user's level of interest in the service is symbolically expressed as follows:
- Exit - user ignored the recommendation
- Simple interest - user clicked the recommended service
- Interest - user clicked on one of the service's offers or proceeded to order the service
- Order - user ordered the service

 Following that, the rewards are mapped to scalars, combined, then normalized to be between 0 and 1