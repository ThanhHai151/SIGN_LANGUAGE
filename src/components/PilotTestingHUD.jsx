import React from 'react';
import ControlledPilotV3HUD, { VIETNAMESE_ALPHABET_29 } from './ControlledPilotV3HUD';

/**
 * PilotTestingHUD (Controlled Pilot V3 Compatibility Forwarder).
 * Rebuilt completely from scratch to replace the obsolete Pilot HUD V2.
 */
export default function PilotTestingHUD(props) {
  return <ControlledPilotV3HUD {...props} />;
}

export { VIETNAMESE_ALPHABET_29 };
